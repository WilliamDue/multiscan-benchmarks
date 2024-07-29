#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>
#include "sps.cu.h"

struct Add {
    __device__ inline int operator()(int a, int b) const {
        return a + b;
    }
};

int gpuAssert(cudaError_t code) {
    if(code != cudaSuccess) {
        printf("GPU Error: %s\n", cudaGetErrorString(code));
        return -1;
    }
    return 0;
}

void info() {
    cudaDeviceProp prop;
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices != 0);
    cudaGetDeviceProperties(&prop, 0);
    uint32_t max_hwdth = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    uint32_t max_block = prop.maxThreadsPerBlock;
    uint32_t max_shmen = prop.sharedMemPerBlock;

    printf("Number of devices: %i\n", nDevices);
    printf("Device name: %s\n", prop.name);
    printf("Number of hardware threads: %d\n", max_hwdth);
    printf("Max block size: %d\n", max_block);
    printf("Shared memory size: %d\n", max_shmen);
    puts("====");
}

template<typename T, typename I, typename OP, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
scanBlocks(T* d_in,
           T* d_out,
           OP op,
           T ne,
           const I size) {
    volatile __shared__ T block[ITEMS_PER_THREAD * BLOCK_SIZE];
	volatile __shared__ T block_aux[BLOCK_SIZE];
    I glb_offs = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;

    glbToShmemCpy<T, I, ITEMS_PER_THREAD>(glb_offs, size, ne, d_in, block);

    scanBlock<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, op, ne);
    
    shmemToGlbCpy<T, I, ITEMS_PER_THREAD>(glb_offs, size, d_out, block);
}

void testBlocks(uint32_t size) {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t ITEMS_PER_THREAD = 4;
    const uint32_t GRID_SIZE = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const uint32_t ARRAY_BYTES = size * sizeof(int);

    std::vector<int> h_in(size);
    std::vector<int> h_out(size, 0);

    for (uint32_t i = 0; i < size; ++i) {
        h_in[i] = rand() % 10;
    }

    int *d_in, *d_out;
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));

    gpuAssert(cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice));

    Add op = Add();
    scanBlocks<int, uint32_t, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, op, 0, size);
    cudaDeviceSynchronize();

    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    int acc = 0;
    bool test_passes = true;

    for (uint32_t i = 0; i < size; ++i) {
        if (i % (BLOCK_SIZE * ITEMS_PER_THREAD) == 0) {
            acc = h_in[i];
        } else {
            acc += h_in[i];
        }
        test_passes &= h_out[i] == acc;
    }

    if (test_passes) {
        std::cout << "Block Scan Test Passed using size=" << size << std::endl;
    } else {
        std::cout << "Block Scan Test Failed using size=" << size << std::endl;
    }

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
}

template<typename T, typename I, typename OP, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
scan(T* d_in,
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

    scanBlock<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, op, ne);

    decoupledLookbackScan<T, I, OP, ITEMS_PER_THREAD>(states, block, op, ne, dyn_idx);

    shmemToGlbCpy<T, I, ITEMS_PER_THREAD>(glb_offs, size, d_out, block);
    
}

void testScan(uint32_t size) {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t ITEMS_PER_THREAD = 2;
    const uint32_t NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const uint32_t ARRAY_BYTES = size * sizeof(int);
    const uint32_t STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<int>);

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
    scan<int, uint32_t, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, d_states, size, op, 0, d_dyn_idx_ptr);
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
    info();

    testBlocks(1 << 6);
    testBlocks(1 << 16);
    testBlocks(1 << 26);

    testBlocks(1000);
    testBlocks(100000);
    testBlocks(10000000);

    testScan(1 << 8);
    testScan(1 << 16);
    testScan(1 << 26);

    testScan(1000);
    testScan(100000);
    testScan(10000000);

    return 0;
}