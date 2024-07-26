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

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline T
scanThread(T* shmem,
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
addAuxBlockScan(T* shmem,
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
scanBlock(T* shmem,
          volatile T* shmem_aux,
          OP op,
          const T ne) {
    scanThread<T, I, OP, ITEMS_PER_THREAD>(shmem, shmem_aux, op, ne);

    scanBlock<T, I, OP>(shmem_aux, op);
    
    addAuxBlockScan<T, I, OP, ITEMS_PER_THREAD>(shmem, shmem_aux, op);
}

template<typename I>
__device__ inline I dynamicIndex(I* dyn_idx_ptr) {
    __shared__ I dyn_idx;
    
    if (threadIdx.x == 0)
        dyn_idx = atomicAdd(dyn_idx_ptr, 1);
    
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

template<typename T>
struct Pair {
    T value;
    uint8_t status_and_flag;

    __host__ __device__ Pair(State<T> state, T ne) {
        bool is_aggregate = state.status == Aggregate;
        bool is_prefix = state.status == Prefix;
        status_and_flag = !is_aggregate | (state.status << 1);

        value = ne;
        if (is_aggregate) {
            value = state.aggregate;
        } else if (is_prefix) {
            value = state.prefix;
        }
    }

    __host__ __device__ Pair(T value, Status status, bool flag) : value(value) {
        status_and_flag = flag | (status << 1);
    }

    __host__ __device__ Pair(T value, uint8_t status_and_flag) : value(value), status_and_flag(status_and_flag) { }

    __host__ __device__ Pair(T ne) : value(ne) {
        status_and_flag = false | (Aggregate << 1);
    }

    __host__ __device__ Pair(const Pair<T>& other) : value(other.value), status_and_flag(other.status_and_flag) { }

    __host__ __device__ Pair(const volatile Pair<T>& other) : value(other.value), status_and_flag(other.status_and_flag) { }

    __host__ __device__ bool inline GetFlag() const volatile {
        return status_and_flag & 1;
    }

    __host__ __device__ Status inline GetStatus() const volatile {
        return static_cast<Status>(status_and_flag >> 1);
    }

    __host__ __device__ inline Pair<T>& operator=(const Pair<T>& other) {
        if (this != &other) {
            value = other.value;
            status_and_flag = other.status_and_flag;
        }
        return *this;
    }

    __host__ __device__ inline volatile Pair<T>& operator=(const Pair<T>& other) volatile {
        if (this != &other) {
            value = other.value;
            status_and_flag = other.status_and_flag;
        }
        return *this;
    }
};

__device__ inline Status
combine(Status a, Status b) {
    if (b == Aggregate)
        return a;
    return b;
}


template<typename T, typename I, typename OP>
__device__ inline Pair<T>
scanWarp(volatile Pair<T>* shmem,
         OP op,
         const uint8_t lane) {
    uint8_t h;

    #pragma unroll
    for (uint8_t d = 0; d < LG_WARP; d++) {
        if ((h = 1 << d) <= lane) {
            Pair<T> a = shmem[threadIdx.x - h];
            Pair<T> b = shmem[threadIdx.x];
            bool new_flag = a.GetFlag() || b.GetFlag();
            Status new_status = combine(a.GetStatus(), b.GetStatus());
            shmem[threadIdx.x] = Pair<T>(b.GetFlag() ? b.value : op(a.value, b.value), new_status, new_flag);
        }
    }
    return shmem[threadIdx.x];
}

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
decoupledLookbackScan(State<T>* states,
                      T* shmem,
                      OP op,
                      const T ne,
                      I dyn_idx) {
    volatile __shared__ uint8_t _shmem_states[WARP * sizeof(Pair<T>)];
    volatile Pair<T>* shmem_states = reinterpret_cast<volatile Pair<T>*>(&_shmem_states[0]);
    volatile __shared__ T shmem_prefix;
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const bool is_first = threadIdx.x == 0;

    T aggregate = shmem[ITEMS_PER_THREAD * blockDim.x - 1];
    states[dyn_idx].aggregate = aggregate;
    
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].prefix = aggregate;
    } else if (is_first) {
        states[dyn_idx].prefix = ne;
    }
    
    __threadfence();
    if (dyn_idx == 0 && is_first) {
        states[dyn_idx].status = Prefix;
    } else if (is_first) {
        states[dyn_idx].status = Aggregate;
    }

    __syncthreads();

    T prefix = ne;
    if (threadIdx.x < WARP && dyn_idx != 0) {
        I lookback_idx = dyn_idx - threadIdx.x - 1;
        I lookback_warp = 0;

        do {
            if (threadIdx.x < dyn_idx && lookback_warp <= lookback_idx) {
                shmem_states[threadIdx.x] = Pair<T>(states[lookback_idx - lookback_warp], ne);
            } else {
                shmem_states[threadIdx.x] = Pair<T>(ne);
            }

            T result = scanWarp<T, I, OP>(shmem_states, op, lane).value;

            if (shmem_states[WARP - 1].GetStatus() == Invalid)
                continue;
                
            if (is_first)
                prefix = op(result, prefix);
            
            lookback_warp -= WARP;
        } while (shmem_states[WARP - 1].GetStatus() != Prefix);
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

