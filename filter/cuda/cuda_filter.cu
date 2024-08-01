#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"

struct Add {
    __device__ inline int operator()(int a, int b) const {
        return a + b;
    }
};

template<typename T, typename I, typename OP, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
spsScan(T* d_in,
     T* d_out,
     volatile State<T>* states,
     I size,
     OP op,
     const T ne,
     volatile I* dyn_idx_ptr) {
    volatile __shared__ T block[ITEMS_PER_THREAD * BLOCK_SIZE];
	volatile __shared__ T block_aux[BLOCK_SIZE];
    
    I dyn_idx = dynamicIndex<I>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    glbToShmemCpy<T, I, ITEMS_PER_THREAD>(glb_offs, size, ne, d_in, block);

    scan<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, states, op, ne, dyn_idx);

    shmemToGlbCpy<T, I, ITEMS_PER_THREAD>(glb_offs, size, d_out, block);
    
}

void testScan(uint32_t size) {
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t ITEMS_PER_THREAD = 30;
    const uint32_t NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const uint32_t ARRAY_BYTES = size * sizeof(int);
    const uint32_t STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<int>);
    const uint32_t WARMUP_RUNS = 1000;
    const uint32_t RUNS = 10;

    std::vector<int> h_in(size);
    std::vector<int> h_out(size, 0);

    for (uint32_t i = 0; i < size; ++i) {
        h_in[i] = rand() % 10;
    }

    uint32_t* d_dyn_idx_ptr;
    int *d_in, *d_out;
    State<int>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));

    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));

    Add op = Add();
    
    for (uint32_t i = 0; i < WARMUP_RUNS; ++i) {
        spsScan<int, uint32_t, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, op, 0, d_dyn_idx_ptr);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (uint32_t i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        spsScan<int, uint32_t, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, op, 0, d_dyn_idx_ptr);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    }

    compute_descriptors(temp, RUNS, ARRAY_BYTES);
    free(temp);

    spsScan<int, uint32_t, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, op, 0, d_dyn_idx_ptr);
    cudaDeviceSynchronize();

    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    int acc = 0;

    bool test_passes = true;

    for (uint32_t i = 0; i < size; ++i) {
        acc += h_in[i];
        test_passes &= h_out[i] == acc;
    }

    if (test_passes) {
        std::cout << "Scan Test Passed using size=" << size << std::endl;
    } else {
        std::cout << "Scan Test Failed using size=" << size << std::endl;
    }

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
}

int main() {
    testScan(1 << 8);
    std::cout << "\n";
    testScan(1 << 16);
    std::cout << "\n";
    testScan(1 << 26);
    std::cout << "\n";

    testScan(1000);
    std::cout << "\n";
    testScan(100000);
    std::cout << "\n";
    testScan(100000000);
    std::cout << std::flush;

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
