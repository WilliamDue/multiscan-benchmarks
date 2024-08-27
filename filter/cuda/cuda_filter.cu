#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"

template<typename I>
struct Add {
    __device__ __forceinline__ I operator()(I a, I b) const {
        return a + b;
    }
};


struct Predicate {
    __device__ inline bool operator()(int32_t a) const {
        return 0 < a;
    }
};

template<typename T>
struct Identity {
    __device__ inline T operator()(T a) const {
        return a;
    }
};

template<typename T, typename I, typename PRED, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
filter(T* d_in,
       T* d_out,
       volatile State<I>* states,
       I size,
       I num_logical_blocks,
       PRED pred,
       volatile uint32_t* dyn_idx_ptr,
       volatile I* new_size) {
    volatile __shared__ I block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I block_aux[BLOCK_SIZE];
    T elems[ITEMS_PER_THREAD];
    uint32_t bools = 0;

    uint32_t dyn_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            elems[i] = d_in[gid];
            bool temp = pred(elems[i]);
            bools |= temp << i;
            block[lid] = temp;
        } else {
            block[lid] = 0;
        }
    }
    __syncthreads();

    scan<I, I, Add<I>, ITEMS_PER_THREAD>(block, block_aux, states, Add<I>(), 0, dyn_idx);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((bools >> i) & 1)) {
            d_out[block[lid] - 1] = elems[i];
        }
    }
    
    if (dyn_idx == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1];
    }
    __syncthreads();
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline T
decoupledLookbackScanNoWrite(volatile State<T>* states,
                             volatile T* shmem,
                             OP op,
                             const T ne,
                             uint32_t dyn_idx) {
    volatile __shared__ T values[WARP];
    volatile __shared__ Status statuses[WARP];
    volatile __shared__ T shmem_prefix;
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const bool is_first = threadIdx.x == 0;

    T aggregate = shmem[ITEMS_PER_THREAD * blockDim.x - 1];

    if (is_first) {
        states[dyn_idx].aggregate = aggregate;
    }
    
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].prefix = aggregate;
    }
    
    __threadfence();
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].status = Prefix;
    } else if (is_first) {
        states[dyn_idx].status = Aggregate;
    }

    T prefix = ne;
    if (threadIdx.x < WARP && dyn_idx != 0) {
        I lookback_idx = threadIdx.x + dyn_idx;
        I lookback_warp = WARP;
        Status status = Aggregate;
        do {
            if (lookback_warp <= lookback_idx) {
                I idx = lookback_idx - lookback_warp;
                status = states[idx].status;
                statuses[threadIdx.x] = status;
                values[threadIdx.x] = status == Prefix ? states[idx].prefix : states[idx].aggregate;
            } else {
                statuses[threadIdx.x] = Aggregate;
                values[threadIdx.x] = ne;
            }

            scanWarp<T, I, OP>(values, statuses, op, lane);

            T result = values[WARP - 1];
            status = statuses[WARP - 1];

            if (status == Invalid)
                continue;
                
            if (is_first) {
                prefix = op(result, prefix);
            }

            lookback_warp += WARP;
        } while (status != Prefix);
    }

    if (is_first) {
        shmem_prefix = prefix;
    }

    __syncthreads();

    if (is_first) {
        states[dyn_idx].prefix = op(prefix, aggregate);
        __threadfence();
        states[dyn_idx].status = Prefix;
    }
    
    return shmem_prefix;
}

template<typename T, typename I, typename PRED, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
filterFewerShmemWrite(T* d_in,
                      T* d_out,
                      volatile State<I>* states,
                      I size,
                      I num_logical_blocks,
                      PRED pred,
                      volatile uint32_t* dyn_idx_ptr,
                      volatile I* new_size) {
    volatile __shared__ I block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I block_aux[BLOCK_SIZE];
    T elems[ITEMS_PER_THREAD];
    uint32_t bools = 0;

    uint32_t dyn_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            elems[i] = d_in[gid];
            bool temp = pred(elems[i]);
            bools |= temp << i;
            block[lid] = temp;
        } else {
            block[lid] = 0;
        }
    }
    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(block, block_aux, Add<I>());

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(states, block, Add<I>(), I(), dyn_idx);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((bools >> i) & 1)) {
            d_out[Add<I>()(prefix, block[lid]) - 1] = elems[i];
        }
    }
    
    if (dyn_idx == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, block[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    }
    __syncthreads();
}

template<typename T, typename I, typename PRED, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
filterCoalescedWrite(T* d_in,
                     T* d_out,
                     volatile State<I>* states,
                     I size,
                     I num_logical_blocks,
                     PRED pred,
                     volatile uint32_t* dyn_idx_ptr,
                     volatile I* new_size) {
    volatile __shared__ I block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I block_aux[BLOCK_SIZE];
    volatile __shared__ I block_keep_size;
    T elems[ITEMS_PER_THREAD];
    uint32_t bools = 0;
    I local_offsets[ITEMS_PER_THREAD];

    uint32_t dyn_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            elems[i] = d_in[gid];
            bool temp = pred(elems[i]);
            bools |= temp << i;
            block[lid] = temp;
        } else {
            block[lid] = I();
        }
    }
    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(block, block_aux, Add<I>());

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        local_offsets[i] = block[lid];
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_keep_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1];
    }
    __syncthreads();

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(states, block, Add<I>(), I(), dyn_idx);

    T *block_cast = (T*) &block;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        if ((bools >> i) & 1) {
            block_cast[local_offsets[i] - 1] = elems[i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        if (lid < block_keep_size) {
            elems[i] = block_cast[lid];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        if ((bools >> i) & 1) {
            block[local_offsets[i] - 1] = local_offsets[i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (lid < block_keep_size) {
            d_out[Add<I>()(prefix, block[lid]) - 1] = elems[i];
        }
    }
    
    if (dyn_idx == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, block[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    }
    __syncthreads();
}

void testFilter(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 10;
    assert(ITEMS_PER_THREAD <= 32);

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    uint32_t* d_dyn_idx_ptr;
    I* d_new_size;
    int32_t *d_in, *d_out;
    State<I>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    Predicate pred = Predicate();
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filter<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        filter<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    free(temp);

    filter<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << "\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Test Failed: Due to elements mismatch at index=" << i << "\n";
            }
        } 
    }

    if (test_passes) {
        std::cout << "Filter test passed." << "\n";
    }

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_new_size));
}

void testFilterCoalescedWrite(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 10;
    assert(ITEMS_PER_THREAD <= 32);

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    uint32_t* d_dyn_idx_ptr;
    I* d_new_size;
    int32_t *d_in, *d_out;
    State<I>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    Predicate pred = Predicate();
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filterCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        filterCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    free(temp);

    filterCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Coalesced Write Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << "\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Coalesced Write Test Failed: Due to elements mismatch at index=" << i << "\n";
            }
        } 
    }

    if (test_passes) {
        std::cout << "Filter Coalesced Write test passed." << "\n";
    }

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_new_size));
}

void testFilterFewerShmemWrite(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 10;
    assert(ITEMS_PER_THREAD <= 32);

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    uint32_t* d_dyn_idx_ptr;
    I* d_new_size;
    int32_t *d_in, *d_out;
    State<I>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    Predicate pred = Predicate();
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filterFewerShmemWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        filterFewerShmemWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    free(temp);

    filterFewerShmemWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter With Fewer Shared Memory Writes Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << "\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter With Fewer Shared Memory Writes Test Failed: Due to elements mismatch at index=" << i << "\n";
            }
        } 
    }

    if (test_passes) {
        std::cout << "Filter With Fewer Shared Memory Writes test passed." << "\n";
    }

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_new_size));
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    size_t input_size;
    int32_t* input = read_i32_array(argv[1], &input_size);
    size_t expected_size;
    int32_t* expected = read_i32_array(argv[2], &expected_size);
    testFilter(input, input_size, expected, expected_size);
    printf("\n");
    testFilterCoalescedWrite(input, input_size, expected, expected_size);
    printf("\n");
    testFilterFewerShmemWrite(input, input_size, expected, expected_size);
    free(input);
    free(expected);

    std::cout << std::flush;

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
