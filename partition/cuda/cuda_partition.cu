#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"
#include <cub/cub.cuh>
#define PAD "%-38s "

template<typename T>
struct Tuple {
    T first;
    T second;

    __host__ __device__
    Tuple(T a, T b) : first(a), second(b) {}
    
    __host__ __device__
    Tuple() : first(T()), second(T()) {}

    __host__ __device__
    Tuple(const Tuple<T>& other) : first(other.first), second(other.second) {}

    __host__ __device__
    Tuple(const volatile Tuple<T>& other) : first(other.first), second(other.second) {}
    
    __host__ __device__
    Tuple<T>& operator=(const Tuple<T>& other) {
        if (this != &other) {
            first = other.first;
            second = other.second;
        }
        return *this;
    }

    __host__ __device__
    Tuple<T>& operator=(const volatile Tuple<T>& other) volatile {
        if (this != &other) {
            first = other.first;
            second = other.second;
        }
        return *const_cast<Tuple<T>*>(this);
    }

    __host__ __device__
    Tuple<T>& operator=(const Tuple<T>& other) volatile {
        if (this != &other) {
            first = other.first;
            second = other.second;
        }
        return *const_cast<Tuple<T>*>(this);
    }
};

template<typename I>
struct AddTuple {
    __device__ inline Tuple<I> operator()(Tuple<I> a, Tuple<I> b) const {
        return Tuple<I>(a.first + b.first, a.second + b.second);
    }
};

struct Predicate {
    __device__ inline bool operator()(int32_t a) const {
        return 0 < a;
    }
};

template<typename T, typename I, typename PRED, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
partition(T* d_in,
          T* d_out,
          volatile State<Tuple<I>>* states,
          I size,
          I num_logical_blocks,
          PRED pred,
          volatile uint32_t* dyn_idx_ptr,
          volatile I* offset) {
    volatile __shared__ Tuple<I> block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ Tuple<I> block_aux[BLOCK_SIZE];
    T elems[ITEMS_PER_THREAD];
    uint32_t bools = 0;
    I glb_offset = *offset;

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
            block[lid].first = temp;
            block[lid].second = !temp;
        } else {
            block[lid].first = I();
            block[lid].second = I();
        }
    }
    __syncthreads();

    scan<Tuple<I>, I, AddTuple<I>, ITEMS_PER_THREAD>(block, block_aux, states, AddTuple<I>(), Tuple<I>(), dyn_idx);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((bools >> i) & 1)) {
            d_out[block[lid].first - 1] = elems[i];
        } else if (gid < size && !((bools >> i) & 1)) {
            d_out[block[lid].second + glb_offset - 1] = elems[i];
        }
    }
    
    __syncthreads();
}

template<typename T, typename I, typename PRED, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
partitionCoalescedWrite(T* d_in,
                        T* d_out,
                        volatile State<Tuple<I>>* states,
                        I size,
                        I num_logical_blocks,
                        PRED pred,
                        volatile uint32_t* dyn_idx_ptr,
                        volatile I* offset) {
    volatile __shared__ Tuple<I> block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ Tuple<I> block_aux[BLOCK_SIZE];
    T elems[ITEMS_PER_THREAD];
    uint32_t bools = 0;
    uint32_t is_valid = 0;
    I local_offsets[ITEMS_PER_THREAD];
    I glb_offset = *offset;

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
            is_valid |= 1 << i;
            block[lid].first = temp;
            block[lid].second = !temp;
        } else {
            block[lid].first = I();
            block[lid].second = I();
        }
    }
    __syncthreads();

    scanBlock<Tuple<I>, I, AddTuple<I>, ITEMS_PER_THREAD>(block, block_aux, AddTuple<I>());

    I first_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1].first;
    I second_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1].second;
    I full_size = first_size + second_size;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        if (((bools >> i) & 1)) {
            local_offsets[i] = block[lid].first - 1;
        } else if (((is_valid >> i) & 1)) {
            local_offsets[i] = block[lid].second + first_size - 1;
        }
    }

    __syncthreads();

    Tuple<I> prefix = decoupledLookbackScanNoWrite<Tuple<I>, I, AddTuple<I>, ITEMS_PER_THREAD>(states, block, AddTuple<I>(), Tuple<I>(), dyn_idx);

    T *block_cast_elem = (T*) &block;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_valid >> i) & 1) {
            block_cast_elem[local_offsets[i]] = elems[i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        if (lid < full_size) {
            elems[i] = block_cast_elem[lid];
        }
    }
    __syncthreads();

    I *block_cast_idx = (I*) &block;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        if ((is_valid >> i) & 1) {
            block_cast_idx[local_offsets[i]] = local_offsets[i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        if (lid < first_size) {
            d_out[prefix.first + block_cast_idx[lid]] = elems[i];
        } else if (lid < full_size) {
            d_out[glb_offset + prefix.second + block_cast_idx[lid] - first_size] = elems[i];
        }
    }
    
    __syncthreads();
}

void testPartition(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 15;
    assert(ITEMS_PER_THREAD <= 32);
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<Tuple<I>>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 500;

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }

    uint32_t* d_dyn_idx_ptr;
    I *d_offset;
    int32_t *d_in, *d_out;
    State<Tuple<I>> *d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_offset, sizeof(I)));
    cudaMemset(d_offset, 0, sizeof(I));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    Predicate pred;
    cub::TransformInputIterator<I, Predicate, int*> itr(d_in, pred);
    cudaDeviceSynchronize();
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
    cudaDeviceSynchronize();
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    float * temp = (float *) malloc(sizeof(float) * (WARMUP_RUNS + RUNS));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
        cudaDeviceSynchronize();
        partition<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
        cudaDeviceSynchronize();
        partition<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    gpuAssert(cudaPeekAtLastError());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    size_t moved_bytes = 3 * ARRAY_BYTES;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
    cudaDeviceSynchronize();
    partition<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    
    bool test_passes = true;
    for (I i = 0; i < size; ++i) {
        test_passes &= h_out[i] == expected[i];

        if (!test_passes) {
            std::cout << "Partition Test Failed: Due to elements mismatch at index=" << i << "." << std::endl;
            break;
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, moved_bytes);
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_offset));
}

void testPartitionCoalescedWrite(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 15;
    assert(ITEMS_PER_THREAD <= 32);
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<Tuple<I>>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 500;

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (I i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }

    uint32_t* d_dyn_idx_ptr;
    I *d_offset;
    int32_t *d_in, *d_out;
    State<Tuple<I>> *d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_offset, sizeof(I)));
    cudaMemset(d_offset, 0, sizeof(I));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    Predicate pred;
    cub::TransformInputIterator<I, Predicate, int*> itr(d_in, pred);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
        cudaDeviceSynchronize();
        partitionCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
        cudaDeviceSynchronize();
        partitionCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    size_t moved_bytes = 3 * ARRAY_BYTES;
    

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
    cudaDeviceSynchronize();
    partitionCoalescedWrite<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    
    bool test_passes = true;
    for (I i = 0; i < size; ++i) {
        test_passes &= h_out[i] == expected[i];

        if (!test_passes) {
            std::cout << "Partition Coalesced Write Test Failed: Due to elements mismatch at index=" << i << std::endl;
            break;
        }
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, moved_bytes);
    }
    
    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_offset));
}

void testPartitionCUB(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
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

    I *d_offset;
    int32_t *d_in, *d_out;
    gpuAssert(cudaMalloc((void**)&d_offset, sizeof(I)));
    cudaMemset(d_offset, 0, sizeof(I));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    Predicate pred;
    cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_offset, size, pred);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    float * temp = (float *) malloc(sizeof(float) * RUNS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_offset, size, pred);
        cudaDeviceSynchronize();
        gpuAssert(cudaPeekAtLastError());
    }

    for (I i = 0; i < RUNS; ++i) {
        cudaEventRecord(start, 0);
        cub::DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_offset, size, pred);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(temp + i, start, stop);
        gpuAssert(cudaPeekAtLastError());
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    size_t moved_bytes = 2 * ARRAY_BYTES;

    compute_descriptors(temp, RUNS, moved_bytes);

    free(temp);
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_offset));
    gpuAssert(cudaFree(d_temp_storage));
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    size_t input_size;
    int32_t* input = read_i32_array(argv[1], &input_size);
    size_t expected_size;
    int32_t* expected = read_i32_array(argv[2], &expected_size);
    printf("%s:\n", argv[1]);
    printf(PAD, "Partition:");
    testPartition(input, input_size, expected, expected_size);
    printf(PAD, "Partition Coalesced Write:");
    testPartitionCoalescedWrite(input, input_size, expected, expected_size);
    printf(PAD, "Partition (CUB):");   
    testPartitionCUB(input, input_size, expected, expected_size);
    free(input);
    free(expected);

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
