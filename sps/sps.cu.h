#include <cuda_runtime.h>
#include <cstdint>

const uint8_t LG_WARP = 5;
const uint8_t WARP = 1 << LG_WARP;

template<typename T, typename I, I ITEMS_PER_THREAD>
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

template<typename T, typename I, I ITEMS_PER_THREAD>
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

template<typename T, typename I, typename Func, I ITEMS_PER_THREAD>
__device__ inline void
shmemToGlbCpy(const I glb_offs,
              const I size,
              T* d_write,
              volatile T* shmem_read,
              Func f,
              volatile I* glb_indices) {
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size)
            d_write[glb_indices[gid]] = f(shmem_read[lid]);
    }
    __syncthreads();
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline T
scanThread(volatile T* shmem,
           volatile T* shmem_aux,
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

template<typename T, typename I, typename OP>
__device__ inline T
scanWarp(volatile T* shmem,
         OP op,
         const uint8_t lane) {
    uint8_t h;

    #pragma unroll
    for (uint8_t d = 0; d < LG_WARP; d++)
        if ((h = 1 << d) <= lane)
            shmem[threadIdx.x] = op(shmem[threadIdx.x - h], shmem[threadIdx.x]);
    
    return shmem[threadIdx.x];
}

template<typename T, typename I, typename OP>
__device__ inline T
scanBlock(volatile T* shmem,
          OP op) {
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const I warpid = threadIdx.x >> LG_WARP;

    T res = scanWarp<T, I, OP>(shmem, op, lane);
    __syncthreads();

    if (lane == (WARP - 1))
        shmem[warpid] = res;
    __syncthreads();

    if (warpid == 0)
        scanWarp<T, I, OP>(shmem, op, lane);
    __syncthreads();

    if (warpid > 0)
        res = op(shmem[warpid-1], res);
    __syncthreads();

    shmem[threadIdx.x] = res;
    __syncthreads();
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
addAuxBlockScan(volatile T* shmem,
                volatile T* shmem_aux,
                OP op) {
    if (threadIdx.x > 0) {
        const I offset = threadIdx.x * ITEMS_PER_THREAD;
        const I upper = offset + ITEMS_PER_THREAD;
        const T val = shmem_aux[threadIdx.x - 1];
        #pragma unroll
        for (I lid = offset; lid < upper; lid++) {
            shmem[lid] = op(shmem[lid], val);
        }
    }
	__syncthreads();
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
scanBlock(volatile T* shmem,
          volatile T* shmem_aux,
          OP op,
          const T ne) {
    scanThread<T, I, OP, ITEMS_PER_THREAD>(shmem, shmem_aux, op, ne);

    scanBlock<T, I, OP>(shmem_aux, op);
    
    addAuxBlockScan<T, I, OP, ITEMS_PER_THREAD>(shmem, shmem_aux, op);
}

template<typename I>
__device__ inline I dynamicIndex(volatile I* dyn_idx_ptr) {
    volatile __shared__ I dyn_idx;
    
    if (threadIdx.x == 0)
        dyn_idx = atomicAdd(const_cast<I*>(dyn_idx_ptr), 1);
    
    __syncthreads();
	return dyn_idx;
}

enum Status: uint8_t {
    Invalid = 0,
    Aggregate = 1,
    Prefix = 2,
};

template<typename T>
struct State {
    T aggregate;
    T prefix;
    Status status = Invalid;
};

/*
Combine is associative since it is the binary operation which resuls in the left most element with
Aggregate as an added identity element. The operation has the following table.

P = Prefix
A = Aggregate
X = Invalid
  | P | A | X
-------------
P | P | P | X
-------------
A | P | A | X
-------------
X | P | X | X

If we map X to False and both P and X to True then we get that this corresponds to the or operation.

P = True
A = False
X = True
  | P | A | X
-------------
P | T | T | T
-------------
A | T | F | T
-------------
X | T | T | T

Meaning this can be used for an irregular segmented scan where we can propegate the last status to
the end of a segment while combining Aggregates with P and X.
*/
__device__ inline Status
combine(Status a, Status b) {
    if (b == Aggregate)
        return a;
    return b;
}

template<typename T, typename I, typename OP>
__device__ inline void
scanWarp(volatile T* values,
         volatile Status* statuses,
         OP op,
         const uint8_t lane) {
    uint8_t h;
    const I tid = threadIdx.x;

    #pragma unroll
    for (uint8_t d = 0; d < LG_WARP; d++) {
        if ((h = 1 << d) <= lane) {
            bool is_not_aggregate = statuses[tid] != Aggregate;
            values[tid] = is_not_aggregate ? values[tid] : op(values[tid - h], values[tid]);
            statuses[tid] = combine(statuses[tid - h], statuses[tid]);
        }
    }
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
decoupledLookbackScan(volatile State<T>* states,
                      volatile T* shmem,
                      OP op,
                      const T ne,
                      I dyn_idx) {
    volatile __shared__ T values[WARP];
    volatile __shared__ Status statuses[WARP];
    volatile __shared__ T shmem_prefix;
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const bool is_first = threadIdx.x == 0;

    T aggregate = shmem[ITEMS_PER_THREAD * blockDim.x - 1];

    states[dyn_idx].aggregate = aggregate;
    
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].prefix = aggregate;
    }
    
    __threadfence();
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].status = Prefix;
    } else if (is_first) {
        states[dyn_idx].status = Aggregate;
    }

        

    T prefix = ne;
    if (threadIdx.x < WARP && dyn_idx != 0) {
        I lookback_idx = threadIdx.x + dyn_idx;
        I lookback_warp = WARP;
        Status status = Aggregate;
        do {
            if (lookback_warp <= lookback_idx) {
                I idx = lookback_idx - lookback_warp;
                status = states[idx].status;
                statuses[threadIdx.x] = status;
                values[threadIdx.x] = status == Prefix ? states[idx].prefix : states[idx].aggregate;
            } else {
                statuses[threadIdx.x] = Aggregate;
                values[threadIdx.x] = ne;
            }

            scanWarp<T, I, OP>(values, statuses, op, lane);

            T result = values[WARP - 1];
            status = statuses[WARP - 1];

            if (status == Invalid)
                continue;
                
            if (is_first) {
                prefix = op(result, prefix);
            }

            lookback_warp += WARP;
        } while (status != Prefix);
    }

    if (is_first) {
        shmem_prefix = prefix;
    }

    __syncthreads();
    prefix = shmem_prefix;

    if (is_first) {
        states[dyn_idx].prefix = op(prefix, aggregate);
        __threadfence();
        states[dyn_idx].status = Prefix;
    }
    
    const I offset = threadIdx.x * ITEMS_PER_THREAD;
    const I upper = offset + ITEMS_PER_THREAD;
    #pragma unroll
    for (I lid = offset; lid < upper; lid++) {
        shmem[lid] = op(prefix, shmem[lid]);
    }
    __syncthreads();
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
scan(volatile T* block,
     volatile T* block_aux,
     volatile State<T>* states,
     OP op,
     const T ne,
     I dyn_idx) {
    scanBlock<T, I, OP, ITEMS_PER_THREAD>(block, block_aux, op, ne);

    decoupledLookbackScan<T, I, OP, ITEMS_PER_THREAD>(states, block, op, ne, dyn_idx);
}