#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>
#include <sys/time.h>
#include "sps.cu.h"

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
    unsigned int resolution = 1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff < 0);
}

void compute_descriptors(timeval* measurements, size_t size, size_t bytes) {   
    double sample_mean = 0.0;
    double sample_variance = 0.0;
    double sample_gbps = 0.0;
    timeval t_diff;
    double diff;
    double d_size = (double) size;
    
    for (size_t i = 0; i < size; i++) {
        t_diff = measurements[i];
        diff = t_diff.tv_sec * 1e6 + t_diff.tv_usec;
        sample_mean += diff / d_size;
        sample_variance += (diff * diff) / d_size;
        sample_gbps += bytes / (500 * d_size * diff);
    }
    double sample_std = sqrt(sample_variance);
    double bound = (0.95 * sample_std) / sqrt(d_size - 1);

    printf("Average time: %.0lfμs", sample_mean);
    printf(" (95%% CI: [%.0lfμs, %.0lfμs])\n", sample_mean - bound, sample_mean + bound);

    
    printf("Measured througput: %.0lfGB/s\n", sample_gbps);
}

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

void benchMemcpy(uint32_t size) {
    const uint32_t WARMUP_RUNS = 50;
    const uint32_t RUNS = 10;
    const uint32_t ARRAY_BYTES = size * sizeof(int);
    int *d_in, *d_out;

    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));

    for (uint32_t i = 0; i < WARMUP_RUNS; ++i) {
        cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (uint32_t i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
    }

    compute_descriptors(temp, RUNS, ARRAY_BYTES);
    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
}

void testScan(uint32_t size) {
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t ITEMS_PER_THREAD = 30;
    const uint32_t NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const uint32_t ARRAY_BYTES = size * sizeof(int);
    const uint32_t STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<int>);
    const uint32_t WARMUP_RUNS = 300;
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
    info();
    
    testBlocks(1 << 6);
    testBlocks(1 << 16);
    testBlocks(1 << 26);

    testBlocks(1000);
    testBlocks(100000);
    testBlocks(10000000);
    std::cout << "\n";

    benchMemcpy(100000000);
    std::cout << "\n";
    
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
