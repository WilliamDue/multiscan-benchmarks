#include <cstdint>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>

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