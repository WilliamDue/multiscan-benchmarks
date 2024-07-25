#include <cuda_runtime.h>
#include <cstdint>

const uint8_t LG_WARP = 5;
const uint8_t WARP = 1 << LG_WARP;

__device__ inline int add(int a, int b) {
    return a + b;
}

template<class T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
glbToShmemCpy(const I glb_offs,
              const I size,
              const T ne,
              T* d_read,
              volatile T* shmem_write) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        shmem_write[lid] = gid < size ? d_read[gid] : ne;
    }
    __syncthreads();
}

template<class T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
shmemToGlbCpy(const I glb_offs,
              const I size,
              T* d_write,
              volatile T* shmem_read) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size)
            d_write[gid] = shmem_read[lid];
    }
    __syncthreads();
}

template<typename T, typename I, typename OP, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__device__ inline T
threadScan(T shmem[ITEMS_PER_THREAD * BLOCK_SIZE],
           volatile T shmem_aux[BLOCK_SIZE],
           OP op,
           const T ne) {
    const I offset = threadIdx.x * ITEMS_PER_THREAD;
    const I upper = offset + ITEMS_PER_THREAD;
    T acc = shmem[offset];
    #pragma unroll
    for (I lid = offset + 1; lid < upper; lid++) {
        T tmp = shmem[lid];
        acc = op(acc, tmp);
        shmem[lid] = acc;
    }
    shmem_aux[threadIdx.x] = acc;
	__syncthreads();
}

template<typename T, typename I, typename OP, I BLOCK_SIZE>
__device__ inline T
warpScan(volatile T shmem[BLOCK_SIZE],
         OP op,
         const uint8_t lane) {
    uint8_t h;

    #pragma unroll
    for (uint8_t d = 0; d < LG_WARP; d++)
        if ((h = 1 << d) <= lane)
            arr[tid] = op(arr[tid - h], arr[tid]);
    
    return arr[tid];
}

template<typename T, typename I, typename OP, I BLOCK_SIZE>
__device__ inline void
blockScan(volatile T shmem_aux[BLOCK_SIZE],
          OP op) {
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const I warpid = threadIdx.x >> LG_WARP;

    T res = warpScan<T, I, OP, BLOCK_SIZE>(shmem_aux, op, lane);
    __syncthreads();

    if (lane == (WARP - 1))
        shmem_aux[warpid] = res;
    __syncthreads();

    if (warpid == 0)
        warpScan<T, I, OP, BLOCK_SIZE>(shmem_aux, op, lane);
    __syncthreads();

    if (warpid > 0)
        shmem_aux[threadIdx.x] = op(shmem[warpid-1], res);
    __syncthreads();
}

template<typename T, typename I, typename OP, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__device__ inline void
blockScan(T shmem[ITEMS_PER_THREAD * BLOCK_SIZE],
          volatile T shmem_aux[BLOCK_SIZE],
          OP op) {
    threadScan<T, I, OP, BLOCK_SIZE, ITEMS_PER_THREAD>(shmem, shmem_aux, op);

    blockScan<T, I, OP, BLOCK_SIZE>(shmem_aux, op);

}
