#include <cstdint>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>
#define gpuAssert(x) _gpuAssert(x, __FILE__, __LINE__)


// https://www.gnu.org/software/libc/manual/html_node/Calculating-Elapsed-Time.html
int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
    tv_usec is certainly positive. */
    int temp_sec = x->tv_sec - y->tv_sec;
    int temp_usec = x->tv_usec - y->tv_usec;
    result->tv_sec = temp_sec;
    result->tv_usec = temp_usec;

    /* Return 1 if result is non-positive. */
    return temp_usec <= 0 || temp_sec <= 0;
}

uint64_t round_div(uint64_t a, uint64_t b) {
    return (a + (b / 2)) / b;
}

// https://en.wikipedia.org/wiki/Integer_square_root
uint64_t isqrt(uint64_t y) {
	uint64_t L = 0;
	uint64_t M;
	uint64_t R = y + 1;

    while (L != R - 1) {
        M = L + ((R - L) / 2);

		if (M * M <= y)
			L = M;
		else
			R = M;
	}

    return L;
}

void f(int long unsigned x) {
    return;
}

void compute_descriptors(timeval* measurements, size_t size, size_t bytes) {   
    double sample_mean = 0;
    double sample_variance = 0;
    double sample_gbps = 0;
    size_t new_size = size;

    double d_size = new_size;
    double d_bytes = bytes;
    for (size_t i = 0; i < size; i++) {
        timeval t_diff = measurements[i];
        int diff = t_diff.tv_sec * 1000000 + t_diff.tv_usec;
        assert(diff >= 0);
        double d_diff = diff;
        sample_mean += d_diff;
        sample_variance += d_diff * d_diff;
        sample_gbps += d_bytes / (d_size * 1e3 * d_diff);
    }
    sample_mean /= d_size;
    sample_variance /= d_size;
    double sample_std = sqrt(sample_variance);
    double bound = 0.95 * (sample_std / sqrt(size));

    printf("%.0lfμs ", sample_mean);
    printf("(95%% CI: [%.1lfμs, %.1lfμs]); ", sample_mean - bound, sample_mean + bound);
    printf("%.0lfGB/s\n",  sample_gbps);
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