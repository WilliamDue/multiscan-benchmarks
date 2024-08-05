#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"
#include <cub/cub.cuh>

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
    bool bools[ITEMS_PER_THREAD];

    uint32_t dyn_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            elems[i] = d_in[gid];
            bools[i] = pred(elems[i]);
            block[lid].first = bools[i];
            block[lid].second = !bools[i];
        } else {
            bools[i] = false;
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
        if (gid < size && bools[i]) {
            d_out[block[lid].first - 1] = elems[i];
        } else if (gid < size && !bools[i]) {
            d_out[block[lid].second + *offset - 1] = elems[i];
        }
    }
    
    __syncthreads();
}

void testPartition(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 15;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<Tuple<I>>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 10;

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
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
        partition<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
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
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
        partition<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    size_t moved_bytes = 2 * ARRAY_BYTES;
    
    compute_descriptors(temp, RUNS, moved_bytes);
    free(temp);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_offset, size);
    partition<int32_t, I, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_offset);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    
    bool test_passes = true;
    for (I i = 0; i < size; ++i) {
        test_passes &= h_out[i] == expected[i];

        if (!test_passes) {
            std::cout << "Partition Test Failed: Due to elements mismatch at index=" << i << std::endl;
            break;
        }
    }

    if (test_passes) {
        std::cout << "Partition test passed." << std::endl;
    }

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_offset));
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    size_t input_size;
    int32_t* input = read_i32_array(argv[1], &input_size);
    size_t expected_size;
    int32_t* expected = read_i32_array(argv[2], &expected_size);
    testPartition(input, input_size, expected, expected_size);
    free(input);
    free(expected);

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
