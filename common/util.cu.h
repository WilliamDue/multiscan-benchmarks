#include <cstdint>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>
#define gpuAssert(x) _gpuAssert(x, __FILE__, __LINE__)

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
    unsigned long resolution = 1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff < 0);
}

long unsigned round_div(unsigned long a, unsigned long b) {
    return (a + (b / 2)) / b;
}

// https://en.wikipedia.org/wiki/Integer_square_root
unsigned long isqrt(unsigned long y) {
	unsigned long L = 0;
	unsigned long M;
	unsigned long R = y + 1;

    while (L != R - 1) {
        M = L + ((R - L) / 2);

		if (M * M <= y)
			L = M;
		else
			R = M;
	}

    return L;
}

void compute_descriptors(timeval* measurements, size_t size, size_t bytes) {   
    unsigned long sample_mean = 0;
    unsigned long sample_variance = 0;
    double sample_gbpms = 0.0;
    timeval t_diff;
    unsigned long diff;
    
    for (size_t i = 0; i < size; i++) {
        t_diff = measurements[i];
        diff = t_diff.tv_sec * 1000000 + t_diff.tv_usec;
        sample_mean += diff;
        sample_variance += diff * diff;
        sample_gbpms += round_div(bytes, diff);
    }
    sample_mean /= size;
    sample_variance /= size;
    unsigned long sample_std = isqrt(sample_variance);
    double bound = (0.95 * sample_std) / isqrt(size);

    printf("%7.0lfμs ", (double) sample_mean);
    printf("(95%% CI: [%7.0lfμs, %7.0lfμs]); ", sample_mean - bound, sample_mean + bound);
    printf("%5.0lfGB/s\n", (double) (sample_gbpms / (size * 1000)));
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