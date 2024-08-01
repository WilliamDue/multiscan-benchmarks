#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"

template<typename I>
struct Add {
    __device__ inline I operator()(I a, I b) const {
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
       volatile I* dyn_idx_ptr,
       volatile I* new_size) {
    volatile __shared__ I block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I block_aux[BLOCK_SIZE];
    T elems[ITEMS_PER_THREAD];
    bool bools[ITEMS_PER_THREAD];

    I dyn_idx = dynamicIndex<I>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            elems[i] = d_in[gid];
            bools[i] = pred(elems[i]);
            block[lid] = bools[i];
        } else {
            bools[i] = false;
            block[lid] = 0;
        }
    }
    __syncthreads();

    scan<I, I, Add<I>, ITEMS_PER_THREAD>(block, block_aux, states, Add<I>(), 0, dyn_idx);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && bools[i]) {
            d_out[block[lid] - 1] = elems[i];
        }
    }
    
    if (dyn_idx == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1];
    }
    __syncthreads();
}

void testFilter(int32_t* input, size_t input_size, int32_t* expected, size_t expected_size) {
    const uint32_t size = input_size;
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t ITEMS_PER_THREAD = 30;
    const uint32_t NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const uint32_t ARRAY_BYTES = size * sizeof(int32_t);
    const uint32_t STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<uint32_t>);
    const uint32_t WARMUP_RUNS = 1000;
    const uint32_t RUNS = 10;

    std::vector<int32_t> h_in(size);
    std::vector<int32_t> h_out(size, 0);

    for (uint32_t i = 0; i < size; ++i) {
        h_in[i] = input[i];
    }
    
    uint32_t* d_dyn_idx_ptr;
    uint32_t* d_new_size;
    int32_t *d_in, *d_out;
    State<uint32_t>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(uint32_t)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    Predicate pred = Predicate();
    
    for (uint32_t i = 0; i < WARMUP_RUNS; ++i) {
        filter<int32_t, uint32_t, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (uint32_t i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        filter<int32_t, uint32_t, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    compute_descriptors(temp, RUNS, ARRAY_BYTES);
    free(temp);

    filter<int32_t, uint32_t, Predicate, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, NUM_LOGICAL_BLOCKS, pred, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    uint32_t temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (uint32_t i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Test Failed: Due to elements mismatch at index=" << i << std::endl;
            }
        } 
    }

    if (test_passes) {
        std::cout << "Filter test passed." << std::endl;
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
    free(input);
    free(expected);

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
