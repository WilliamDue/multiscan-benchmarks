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

int main() {
    info();

    const uint32_t BLOCK_SIZE = 32;
    const uint32_t ITEMS_PER_THREAD = 4;
    const uint32_t SIZE = 1024;
    const uint32_t BLOCKS = (SIZE + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const uint32_t ARRAY_BYTES = SIZE * sizeof(int);

    std::vector<int> h_in(SIZE);
    std::vector<int> h_out(SIZE, 0);

    for (uint32_t i = 0; i < SIZE; ++i) {
        h_in[i] = 1;
    }

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_in.data(), ARRAY_BYTES, cudaMemcpyHostToDevice);

    scanBlocks<int, uint32_t, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<BLOCKS, BLOCK_SIZE>>>(d_in, d_out, Add(), 0, SIZE);

    cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < SIZE; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}