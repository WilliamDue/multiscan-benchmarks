#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"
#include <unistd.h>
#define PAD "%-42s "

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

template<typename T, typename I, typename PRED, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
filterTwoKernel1(T* d_in,
                 I* d_out,
                 I* d_flags_out,
                 volatile State<I>* states,
                 I size,
                 I num_logical_blocks,
                 PRED pred,
                 volatile uint32_t* dyn_idx_ptr) {
    volatile __shared__ I block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I block_aux[BLOCK_SIZE];

    uint32_t dyn_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            T elem = d_in[i];
            I flag = pred(elem);
            block[lid] = flag;
            d_flags_out[gid] = flag;
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
        if (gid < size) {
            d_out[gid] = block[lid];
        }
    }
}

template<typename T, typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
filterTwoKernel2(T* d_in,
                 I* d_flags,
                 I* d_offset,
                 T* d_out,
                 I size) {
    volatile __shared__ T elem_block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I flags_block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I d_offset_block[ITEMS_PER_THREAD * BLOCK_SIZE];
    I glb_offs = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            flags_block[lid] = d_flags[gid];
        } else {
            flags_block[lid] = false;
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            elem_block[lid] = d_in[gid];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            d_offset_block[lid] = d_offset[gid];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && flags_block[lid]) {
            d_out[d_offset_block[lid] - 1] = elem_block[lid];
        }
    }
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
    I block_keep_size;
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

    block_keep_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1];
    
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
    const I RUNS = 500;
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
    
    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filter<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        filter<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    filter<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << ".\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Test Failed: Due to elements mismatch at index=" << i << ".\n";
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_new_size));
}

void testFilterCoalescedWrite(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 512;
    const I ITEMS_PER_THREAD = 14;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 500;
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
    
    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filterCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        filterCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;

    filterCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Coalesced Write Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << ".\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Coalesced Write Test Failed: Due to elements mismatch at index=" << i << ".\n";
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    }

    free(temp);
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
    const I RUNS = 500;
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
    
    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filterFewerShmemWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        filterFewerShmemWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    filterFewerShmemWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter With Fewer Shared Memory Writes Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << ".\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter With Fewer Shared Memory Writes Test Failed: Due to elements mismatch at index=" << i << ".\n";
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_new_size));
}

void testFilterCUB(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 500;
    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    I* d_new_size;
    int32_t *d_in, *d_out;
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    Predicate pred = Predicate();

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_new_size, size, pred);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_new_size, size, pred);
        cudaDeviceSynchronize();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_new_size, size, pred);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;

    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_new_size, size, pred);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter (CUB) Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << ".\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter (CUB) Test Failed: Due to elements mismatch at index=" << i << ".\n";
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_temp_storage));
}

void testFilterTwoKernels(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 4;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I OFFSETS_BYTES = size * sizeof(I);
    const I FLAGS_BYTES = size * sizeof(I);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 500;
    assert(ITEMS_PER_THREAD <= 32);

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);
    std::vector<I> h_offsets(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    uint32_t* d_dyn_idx_ptr;
    I* d_flags;
    I* d_offsets;
    int32_t *d_in, *d_out;
    State<I>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_flags, FLAGS_BYTES));
    gpuAssert(cudaMalloc((void**)&d_offsets, OFFSETS_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    Predicate pred = Predicate();
    
    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        filterTwoKernel1<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            d_in,
            d_offsets,
            d_flags,
            d_states,
            size,
            NUM_LOGICAL_BLOCKS,
            pred,
            d_dyn_idx_ptr
        );
        cudaDeviceSynchronize();
        filterTwoKernel2<int32_t, I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            d_in,
            d_flags,
            d_offsets,
            d_out,
            size
        );
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        filterTwoKernel1<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            d_in,
            d_offsets,
            d_flags,
            d_states,
            size,
            NUM_LOGICAL_BLOCKS,
            pred,
            d_dyn_idx_ptr
        );
        cudaDeviceSynchronize();
        filterTwoKernel2<int32_t, I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            d_in,
            d_flags,
            d_offsets,
            d_out,
            size
        );
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    filterTwoKernel1<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        d_in,
        d_offsets,
        d_flags,
        d_states,
        size,
        NUM_LOGICAL_BLOCKS,
        pred,
        d_dyn_idx_ptr
    );
    cudaDeviceSynchronize();
    filterTwoKernel2<int32_t, I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        d_in,
        d_flags,
        d_offsets,
        d_out,
        size
    );
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_offsets.data(), d_offsets, OFFSETS_BYTES, cudaMemcpyDeviceToHost));
    temp_size = h_offsets[size - 1];
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << ".\n";
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Test Failed: Due to elements mismatch at index=" << i << ".\n";
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, ARRAY_BYTES + temp_size * sizeof(int32_t));
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_flags));
    gpuAssert(cudaFree(d_offsets));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
}


int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    size_t input_size;
    int32_t* input = read_i32_array(argv[1], &input_size);
    size_t expected_size;
    int32_t* expected = read_i32_array(argv[2], &expected_size);
    printf("%s:\n", argv[1]);
    /*
    printf(PAD, "Filter:");
    testFilter(input, input_size, expected, expected_size);
    printf(PAD, "Filter Coalesced Write:");
    testFilterCoalescedWrite(input, input_size, expected, expected_size);
    printf(PAD, "Filter With Fewer Shared Memory Writes:");
    testFilterFewerShmemWrite(input, input_size, expected, expected_size);
    printf(PAD, "Filter (CUB):");
    testFilterCUB(input, input_size, expected, expected_size);
    */
    printf(PAD, "Filter Two Kernels:");
    testFilterTwoKernels(input, input_size, expected, expected_size);
    free(input);
    free(expected);

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
