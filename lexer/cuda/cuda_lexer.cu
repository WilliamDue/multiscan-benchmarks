#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"
#include <math.h>
#define PAD "%-38s "

using token_t = uint8_t;
using state_t = uint16_t;

const uint32_t NUM_STATES = 12;
const uint32_t NUM_TRANS = 256;
// const token_t IGNORE_TOKEN = 0;
const state_t ENDO_MASK = 15;
const state_t ENDO_OFFSET = 0;
const state_t TOKEN_MASK = 112;
const state_t TOKEN_OFFSET = 4;
const state_t ACCEPT_MASK = 128;
const state_t ACCEPT_OFFSET = 7;
const state_t PRODUCE_MASK = 256;
const state_t PRODUCE_OFFSET = 8;
const state_t IDENTITY = 74;

state_t h_to_state[NUM_TRANS] =
        {75, 75, 75, 75, 75, 75, 75, 75, 75, 128, 128, 75, 75, 128,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 128, 75, 75, 75, 75, 75, 75, 75, 161, 178, 75,
         75, 75, 75, 75, 75, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 75, 75, 75, 75, 75, 75, 75, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 75, 75,
         75, 75, 75, 75, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         147, 147, 147, 147, 147, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         75, 75, 75, 75};

state_t h_compose[NUM_STATES * NUM_STATES] =
    {132, 392, 392, 392, 132, 392, 392, 392, 132, 392, 128, 75,
     421, 421, 421, 421, 421, 421, 421, 421, 421, 421, 161, 75,
     438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 178, 75,
     407, 407, 407, 153, 407, 407, 407, 153, 407, 153, 147, 75,
     132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 75,
     421, 421, 421, 421, 421, 421, 421, 421, 421, 421, 421, 75,
     438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 75,
     407, 407, 407, 407, 407, 407, 407, 407, 407, 407, 407, 75,
     392, 392, 392, 392, 392, 392, 392, 392, 392, 392, 392, 75,
     153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 75,
     128, 161, 178, 147, 132, 421, 438, 407, 392, 153, 74, 75,
     75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75};


__device__ __host__ __forceinline__ state_t get_index(state_t state) {
    return (state & ENDO_MASK) >> ENDO_OFFSET;
}

__device__ __host__ __forceinline__ token_t get_token(state_t state) {
    return (state & TOKEN_MASK) >> TOKEN_OFFSET;
}

__device__ bool is_accept(state_t state) {
    return (state & ACCEPT_MASK) >> ACCEPT_OFFSET;
}

__device__ __host__ __forceinline__ bool is_produce(state_t state) {
    return (state & PRODUCE_MASK) >> PRODUCE_OFFSET;
}

struct LexerCtx {
    state_t* d_to_state;
    state_t* d_compose;

    LexerCtx() : d_to_state(NULL), d_compose(NULL) {
        cudaMalloc(&d_to_state, sizeof(h_to_state));
        cudaMemcpy(d_to_state, h_to_state, sizeof(h_to_state),
                cudaMemcpyHostToDevice);
        cudaMalloc(&d_compose, sizeof(h_compose));
        cudaMemcpy(d_compose, h_compose, sizeof(h_compose),
                cudaMemcpyHostToDevice);
    }

    void Cleanup() {
        if (d_to_state) cudaFree(d_to_state);
        if (d_compose) cudaFree(d_compose);
    }

    __device__ __host__ __forceinline__
    state_t operator()(const state_t &a, const state_t &b) const {
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
    }

    __device__ __host__ __forceinline__
    state_t operator()(const volatile state_t &a, const volatile state_t &b) const {
        return d_compose[get_index(b) * NUM_STATES + get_index(a)];
    }

    __device__ __host__ __forceinline__
    state_t to_state(const char &a) const {
        return d_to_state[a];
    }
};

template<typename I>
struct Add {
    __device__ __forceinline__ I operator()(I a, I b) const {
        return a + b;
    }
};

template<typename T, typename I, typename OP, I ITEMS_PER_THREAD>
__device__ inline void
decoupledLookbackScanSuffix(volatile State<T>* states,
                            volatile state_t* suffixes,
                            volatile T* shmem,
                            OP op,
                            const T ne,
                            uint32_t dyn_idx) {
    volatile __shared__ T values[WARP];
    volatile __shared__ Status statuses[WARP];
    volatile __shared__ T shmem_prefix;
    const uint8_t lane = threadIdx.x & (WARP - 1);
    const bool is_first = threadIdx.x == 0;

    T aggregate = shmem[ITEMS_PER_THREAD * blockDim.x - 1];

    if (is_first) {
        states[dyn_idx].aggregate = aggregate;
    }
    
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
    const I offset = threadIdx.x * ITEMS_PER_THREAD;
    const I upper = offset + ITEMS_PER_THREAD;
    #pragma unroll
    for (I lid = offset; lid < upper; lid++) {
        shmem[lid] = op(prefix, shmem[lid]);
    }

    if (is_first) {
        states[dyn_idx].prefix = op(prefix, aggregate);
        suffixes[dyn_idx] = shmem[0]; 
        __threadfence();
        states[dyn_idx].status = Prefix;
    }
    
    __syncthreads();
}

template <typename T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
copyFromGlbToShr(
    const I glb_offs,
    const I num_items,
    const I size,
    T* glb,
    volatile T* shr
) {
    const I NUM_ITEMS_BYTES = min(num_items, size - glb_offs) * sizeof(T);
    const I TOTAL_ITERS = 1 + (NUM_ITEMS_BYTES - 1) / sizeof(uint64_t);
    const I ITERS = 1 + (TOTAL_ITERS - 1) / (ITEMS_PER_THREAD * blockDim.x);
    
    #pragma unroll
    for (I j = 0; j < ITERS; j++) { 
        #pragma unroll
        for (I i = 0; i < ITEMS_PER_THREAD; i++) {
            I lid = j * ITEMS_PER_THREAD * blockDim.x + i * blockDim.x + threadIdx.x;
            I lid_byte = lid * sizeof(uint64_t);
            if (lid_byte + sizeof(uint64_t) < NUM_ITEMS_BYTES) {
                reinterpret_cast<volatile uint64_t*>(shr)[lid] = reinterpret_cast<uint64_t*>(glb + glb_offs)[lid];
            } else {
                #pragma unroll
                for (I k = lid_byte; k < NUM_ITEMS_BYTES; k++) {
                    reinterpret_cast<volatile uint8_t*>(shr)[k] = reinterpret_cast<uint8_t*>(glb + glb_offs)[k];
                }
            }
        }
    }

    __syncthreads();
}

template <typename T, typename I, I ITEMS_PER_THREAD>
__device__ inline void
copyFromShrToGlb(
    const I glb_offs,
    const I num_items,
    const I size,
    volatile T* shr,
    T* glb
) {
    const I NUM_ITEMS_BYTES = min(num_items, size - glb_offs) * sizeof(T);
    const I TOTAL_ITERS = 1 + (NUM_ITEMS_BYTES - 1) / sizeof(uint64_t);
    const I ITERS = 1 + (TOTAL_ITERS - 1) / (ITEMS_PER_THREAD * blockDim.x);

    #pragma unroll
    for (I j = 0; j < ITERS; j++) { 
        #pragma unroll
        for (I i = 0; i < ITEMS_PER_THREAD; i++) {
            I lid = j * ITEMS_PER_THREAD * blockDim.x + i * blockDim.x + threadIdx.x;
            I lid_byte = lid * sizeof(uint64_t);
            if (lid_byte + sizeof(uint64_t) < NUM_ITEMS_BYTES) {
                reinterpret_cast<uint64_t*>(glb + glb_offs)[lid] = reinterpret_cast<volatile uint64_t*>(shr)[lid];
            } else {
                #pragma unroll
                for (I k = lid_byte; k < NUM_ITEMS_BYTES; k++) {
                    reinterpret_cast<uint8_t*>(glb + glb_offs)[k] = reinterpret_cast<volatile uint8_t*>(shr)[k];
                }
            }
        }
    }

    __syncthreads();
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexerTwoKernels1(LexerCtx ctx,
                uint8_t* d_in,
                state_t* d_states_out,
                volatile State<state_t>* state_states,
                I size,
                I num_logical_blocks,
                volatile uint32_t* dyn_index_ptr) {
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t states_aux[BLOCK_SIZE];
    const I REG_MEM = 1 + ((ITEMS_PER_THREAD - 1) / sizeof(uint64_t));
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    states_aux[threadIdx.x] = ctx.to_state(threadIdx.x);

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I uint64_lid = i * blockDim.x + threadIdx.x;
        I lid = sizeof(uint64_t) * uint64_lid;
        I gid = glb_offs + lid;
        if (gid + sizeof(uint64_t) < size) {
            copy_reg[i] = *((uint64_t*) (gid + d_in));
        } else {
            #pragma unroll
            for (I j = 0; j < sizeof(uint64_t); j++) {
                I loc_gid = gid + j;
                if (loc_gid < size) {
                    chars_reg[sizeof(uint64_t) * i + j] = d_in[loc_gid];
                }
            }
        }
    }
    
    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I _gid = glb_offs + sizeof(uint64_t) * lid;
        #pragma unroll
        for (I j = 0; j < sizeof(uint64_t); j++) {
            I gid = _gid + j;
	        I lid_off = sizeof(uint64_t) * lid + j;
	        I reg_off = sizeof(uint64_t) * i + j;
	        bool is_in_block = lid_off < ITEMS_PER_THREAD * BLOCK_SIZE; 
            if (gid < size && is_in_block) {
                states[lid_off] = states_aux[chars_reg[reg_off]];
            } else if (is_in_block) {
                states[lid_off] = IDENTITY;
            }
        }
    }

    __syncthreads();

    scan<state_t, I, LexerCtx, ITEMS_PER_THREAD>(states, states_aux, state_states, ctx, IDENTITY, dyn_index);

    copyFromShrToGlb<state_t, I, ITEMS_PER_THREAD>(glb_offs, ITEMS_PER_THREAD * blockDim.x, size, states, d_states_out);
    /*
    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            d_states_out[gid] = states[lid];
        }
    }
    */
    
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexerTwoKernels2(state_t* d_states_in,
                 uint32_t* d_index_out,
                 token_t* d_token_out,
                 volatile State<I>* index_states,
                 I size,
                 I num_logical_blocks,
                 volatile uint32_t* dyn_index_ptr,
                 volatile I* new_size,
                 volatile bool* is_valid) {
    const I STATES_BYTES = sizeof(state_t) * ITEMS_PER_THREAD * BLOCK_SIZE + 1;
    const I INDICES_AUX_BYTES = sizeof(I) * BLOCK_SIZE;
    volatile __shared__ I indices[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ uint8_t shmem[INDICES_AUX_BYTES < STATES_BYTES ? STATES_BYTES : INDICES_AUX_BYTES];
    volatile I* indices_aux = (volatile I*) shmem;
    volatile state_t* states = (volatile state_t*) shmem;
    token_t tokens[ITEMS_PER_THREAD];
    uint32_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    __syncthreads();

    copyFromGlbToShr<state_t, I, ITEMS_PER_THREAD>(glb_offs, ITEMS_PER_THREAD * BLOCK_SIZE + 1, size, d_states_in, states);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            tokens[i] = get_token(states[lid]);
            temp = gid == size - 1 || is_produce(states[lid + 1]);

            
            if (gid == size - 1) {
                *is_valid = is_accept(states[lid]);
            }
        }
        is_produce_state |= temp << i;
        indices[lid] = temp;
    }


    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, Add<I>());

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(index_states, indices, Add<I>(), I(), dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((is_produce_state >> i) & 1)) {
            I offset = Add<I>()(prefix, indices[lid]) - 1;
            d_index_out[offset] = gid;
            d_token_out[offset] = tokens[i];
        }
    }
    
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    }
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexerWorseCopy(LexerCtx ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE + 1];
    volatile __shared__ I indices[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices_aux[BLOCK_SIZE];
    __shared__ state_t next_block_first_state;
    volatile state_t* states_aux = (volatile state_t*) indices;
    uint32_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    states_aux[threadIdx.x] = ctx.to_state(threadIdx.x);
    
    if (threadIdx.x == I()) {
        next_block_first_state = IDENTITY;
    }

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD + 1; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && lid < ITEMS_PER_THREAD * BLOCK_SIZE + 1) {
            states[lid] = states_aux[d_in[gid]];
        } else if (lid < ITEMS_PER_THREAD * BLOCK_SIZE + 1) {
            states[lid] = IDENTITY;
        }
    }

    __syncthreads();
    
    if (threadIdx.x == I()) {
        next_block_first_state = states[ITEMS_PER_THREAD * BLOCK_SIZE];
    }

    scan<state_t, I, LexerCtx, ITEMS_PER_THREAD>(states, states_aux, state_states, ctx, IDENTITY, dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(states[lid], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= temp << i;
        indices[lid] = temp;
    }

    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, Add<I>());

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(index_states, indices, Add<I>(), I(), dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((is_produce_state >> i) & 1)) {
            I offset = Add<I>()(prefix, indices[lid]) - 1;
            d_index_out[offset] = gid;
            d_token_out[offset] = get_token(states[lid]);
        }
    }
    
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
        *is_valid = is_accept(states[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    }
}

template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexer(LexerCtx ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size,
      volatile bool* is_valid) {
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices_aux[BLOCK_SIZE];
    __shared__ state_t next_block_first_state;
    volatile state_t* states_aux = (volatile state_t*) indices;
    const I REG_MEM = 1 + ITEMS_PER_THREAD / sizeof(uint64_t);
    uint64_t copy_reg[REG_MEM];
    uint8_t *chars_reg = (uint8_t*) copy_reg;
    uint32_t is_produce_state = 0;

    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    states_aux[threadIdx.x] = ctx.to_state(threadIdx.x);
    
    if (threadIdx.x == I()) {
        next_block_first_state = IDENTITY;
    }

    __syncthreads();

    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I uint64_lid = i * blockDim.x + threadIdx.x;
        I lid = sizeof(uint64_t) * uint64_lid;
        I gid = glb_offs + lid;
        if (gid + sizeof(uint64_t) < size) {
            copy_reg[i] = *((uint64_t*) (gid + (uint8_t*) d_in));
        } else {
            for (I j = 0; j < sizeof(uint64_t); j++) {
                I loc_gid = gid + j;
                if (loc_gid < size) {
                    chars_reg[sizeof(uint64_t) * i + j] = d_in[loc_gid];
                }
            }
        }
    }
    
    #pragma unroll
    for (I i = 0; i < REG_MEM; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I _gid = glb_offs + sizeof(uint64_t) * lid;
        for (I j = 0; j < sizeof(uint64_t); j++) {
            I gid = _gid + j;
	        I lid_off = sizeof(uint64_t) * lid + j;
	        I reg_off = sizeof(uint64_t) * i + j;
	        bool is_in_block = lid_off < ITEMS_PER_THREAD * BLOCK_SIZE; 
            if (gid < size && is_in_block) {
                states[lid_off] = states_aux[chars_reg[reg_off]];
            } else if (is_in_block) {
                states[lid_off] = IDENTITY;
            } else if (lid_off == ITEMS_PER_THREAD * BLOCK_SIZE) {
                next_block_first_state = states_aux[chars_reg[reg_off]];
            }
        }
    }

    __syncthreads();

    scan<state_t, I, LexerCtx, ITEMS_PER_THREAD>(states, states_aux, state_states, ctx, IDENTITY, dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        bool temp = false;
        if (gid < size) {
            if (lid == ITEMS_PER_THREAD * BLOCK_SIZE - 1) {
                temp = gid == size - 1 || is_produce(ctx(states[lid], next_block_first_state));
            } else {
                temp = gid == size - 1 || is_produce(states[lid + 1]);
            }
        }
        is_produce_state |= temp << i;
        indices[lid] = temp;
    }

    __syncthreads();

    scanBlock<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, Add<I>());

    I prefix = decoupledLookbackScanNoWrite<I, I, Add<I>, ITEMS_PER_THREAD>(index_states, indices, Add<I>(), I(), dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && ((is_produce_state >> i) & 1)) {
            I offset = Add<I>()(prefix, indices[lid]) - 1;
            d_index_out[offset] = gid;
            d_token_out[offset] = get_token(states[lid]);
        }
    }
    
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = Add<I>()(prefix, indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
        *is_valid = is_accept(states[ITEMS_PER_THREAD * BLOCK_SIZE - 1]);
    }
}

void testLexer(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 31;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 50;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    LexerCtx ctx = LexerCtx();

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * size;
    const I SCAN_READ =  sizeof(state_t) * (size + size / 2); // Lowerbound, it does more work.
    
    lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx,
        d_in,
        d_index_out,
        d_token_out,
        d_state_states,
        d_index_states,
        size,
        NUM_LOGICAL_BLOCKS,
        d_dyn_index_ptr,
        d_new_size,
        d_is_valid
    );
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
    
    bool test_passes = is_valid;

    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }

    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];

            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
                break;
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + SCAN_READ + OUT_WRITE);
    }    

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));


    ctx.Cleanup();
}

void testLexerWorseCopy(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 31;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 50;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    LexerCtx ctx = LexerCtx();

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerWorseCopy<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        lexerWorseCopy<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_index_out,
            d_token_out,
            d_state_states,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * size;
    const I SCAN_READ =  sizeof(state_t) * (size + size / 2); // Lowerbound, it does more work.
    
    lexerWorseCopy<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx,
        d_in,
        d_index_out,
        d_token_out,
        d_state_states,
        d_index_states,
        size,
        NUM_LOGICAL_BLOCKS,
        d_dyn_index_ptr,
        d_new_size,
        d_is_valid
    );
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
    
    bool test_passes = is_valid;

    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }

    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];

            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
                break;
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + SCAN_READ + OUT_WRITE);
    }    

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));

    ctx.Cleanup();
}

void testLexerTwoKernels(uint8_t* input,
               size_t input_size,
               uint32_t* expected_indices,
               token_t* expected_tokens,
               size_t expected_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 31;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I IN_ARRAY_BYTES = size * sizeof(uint8_t);
    const I INDEX_OUT_ARRAY_BYTES = size * sizeof(I);
    const I TOKEN_OUT_ARRAY_BYTES = size * sizeof(token_t);
    const I STATES_OUT_BYTES = size * sizeof(state_t);
    const I STATE_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<state_t>);
    const I INDEX_STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<I>);
    const I WARMUP_RUNS = 500;
    const I RUNS = 50;

    std::vector<token_t> h_token_out(size, 0);
    std::vector<I> h_index_out(size, 0);

    uint32_t* d_dyn_index_ptr;
    I* d_new_size;
    bool* d_is_valid;
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    state_t *d_states_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_is_valid, sizeof(bool)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    cudaMemset(d_is_valid, false, sizeof(bool));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_states_out, STATES_OUT_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    LexerCtx ctx = LexerCtx();

    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexerTwoKernels1<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_states_out,
            d_state_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        lexerTwoKernels2<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            d_states_out,
            d_index_out,
            d_token_out,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        lexerTwoKernels1<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            ctx,
            d_in,
            d_states_out,
            d_state_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr
        );
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        lexerTwoKernels2<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
            d_states_out,
            d_index_out,
            d_token_out,
            d_index_states,
            size,
            NUM_LOGICAL_BLOCKS,
            d_dyn_index_ptr,
            d_new_size,
            d_is_valid
        );
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_OUT_STATES = 2 * STATES_OUT_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * size;
    const I SCAN_READ =  sizeof(state_t) * (size + size / 2); // Lowerbound, it does more work.
    
    lexerTwoKernels1<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        ctx,
        d_in,
        d_states_out,
        d_state_states,
        size,
        NUM_LOGICAL_BLOCKS,
        d_dyn_index_ptr
    );
    cudaDeviceSynchronize();
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    lexerTwoKernels2<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(
        d_states_out,
        d_index_out,
        d_token_out,
        d_index_states,
        size,
        NUM_LOGICAL_BLOCKS,
        d_dyn_index_ptr,
        d_new_size,
        d_is_valid
    );
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    bool is_valid = false;
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&is_valid, d_is_valid, sizeof(bool), cudaMemcpyDeviceToHost));
    
    bool test_passes = is_valid;

    if (!test_passes) {
        std::cout << "Lexer Test Failed: The input given to the lexer does not result in an accepting state." << std::endl;
    }

    test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Lexer Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_index_out[i] == expected_indices[i];
            test_passes &= h_token_out[i] == expected_tokens[i];

            if (!test_passes) {
                std::cout << "Lexer Test Failed: Due to elements mismatch at index=" << i << std::endl;
                break;
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + SCAN_READ + OUT_WRITE + IN_OUT_STATES);
    }    

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_states_out));
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));
    gpuAssert(cudaFree(d_is_valid));


    ctx.Cleanup();
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    size_t input_size;
    uint8_t* input = read_u8_file(argv[1], &input_size);
    uint32_t* expected_indices = NULL;
    size_t expected_indices_size = 0;
    uint8_t* expected_tokens = NULL;
    size_t expected_tokens_size = 0;
    read_tuple_u32_u8_array(argv[2],
                            &expected_indices,
                            &expected_indices_size,
                            &expected_tokens,
                            &expected_tokens_size);
    assert(expected_indices_size == expected_tokens_size);
    /*
    size_t input_size = 14;
    uint8_t input[14] = {'t', 'e', 's', 't', ' ', 't', 'e', 's', 't', ' ', 't', 'e', 's', 't'};
    uint32_t expected_indices[5] = {4, 5, 9, 10, 14};
    uint8_t expected_tokens[5] = {1, 0, 1, 0, 1};
    size_t expected_indices_size = 5;
    */
    printf("%s:\n", argv[1]);
    
    printf(PAD, "Lexer:");
    testLexer(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Worse Copy:");
    testLexerWorseCopy(input, input_size, expected_indices, expected_tokens, expected_indices_size);
    printf(PAD, "Lexer Two Kernels:");
    testLexerTwoKernels(input, input_size, expected_indices, expected_tokens, expected_indices_size);

    free(input);
    free(expected_indices);
    free(expected_tokens);
    gpuAssert(cudaPeekAtLastError());
    return 0;
}
