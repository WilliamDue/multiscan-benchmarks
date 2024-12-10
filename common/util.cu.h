#include <cstdint>
#include <assert.h>
#include <cuda_runtime.h>
#define gpuAssert(x) _gpuAssert(x, __FILE__, __LINE__)

void compute_descriptors(float* measurements, size_t size, size_t bytes) {   
    double sample_mean = 0;
    double sample_variance = 0;
    double sample_gbps = 0;
    double factor = bytes / (1000 * size);
    
    for (size_t i = 0; i < size; i++) {
        double diff = max(1e3 * measurements[i], 0.5);
        sample_mean += diff / size;
        sample_variance += (diff * diff) / size;
        sample_gbps += factor / diff;
    }
    double sample_std = sqrt(sample_variance);
    double bound = (0.95 * sample_std) / sqrt(size);

    printf("%.0lfμs ", sample_mean);
    printf("(95%% CI: [%.1lfμs, %.1lfμs]); ", sample_mean - bound, sample_mean + bound);
    printf("%.0lfGB/s\n", sample_gbps);
}

int _gpuAssert(cudaError_t code, const char *fname, int lineno) {
    if(code != cudaSuccess) {
        printf("GPU Error: %s, File: %s, Line: %i\n", cudaGetErrorString(code), fname, lineno);
        fflush(stdout);
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
    printf("Shared memory size: %d\n\n", max_shmen);
}