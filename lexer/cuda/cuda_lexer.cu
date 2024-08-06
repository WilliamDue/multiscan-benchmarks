#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"
#include <math.h>

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

bool is_accept(state_t state) {
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


template<typename I, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
lexer(LexerCtx *ctx,
      uint8_t* d_in,
      uint32_t* d_index_out,
      token_t* d_token_out,
      volatile State<state_t>* state_states,
      volatile State<I>* index_states,
      I size,
      I num_logical_blocks,
      volatile uint32_t* dyn_index_ptr,
      volatile I* new_size) {
    volatile __shared__ state_t states[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ state_t states_aux[BLOCK_SIZE];
    volatile __shared__ I indices[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ I indices_aux[BLOCK_SIZE];
    bool is_produce_state[ITEMS_PER_THREAD];


    uint32_t dyn_index = dynamicIndex<uint32_t>(dyn_index_ptr);
    I glb_offs = dyn_index * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            states[lid] = ctx->to_state(d_in[gid]);
        } else {
            states[lid] = IDENTITY;
        }
    }

    __syncthreads();

    scan<state_t, I, LexerCtx, ITEMS_PER_THREAD>(states, states_aux, state_states, *ctx, IDENTITY, dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        is_produce_state[i] = gid == I() || is_produce(states[lid]);
        indices[lid] = is_produce_state[i];
    }

    __syncthreads();

    scan<I, I, Add<I>, ITEMS_PER_THREAD>(indices, indices_aux, index_states, Add<I>(), I(), dyn_index);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && is_produce_state[i]) {
            I offset = indices[lid] - 1;
            d_index_out[offset] = gid;
            d_token_out[offset] = get_token(states[lid]);
        }
    }
    
    if (dyn_index == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = indices[ITEMS_PER_THREAD * BLOCK_SIZE - 1];
    }

    __syncthreads();
}

void testLexer(uint8_t* input, size_t input_size) {
    using I = uint32_t;
    const I size = input_size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 30;
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
    uint8_t *d_in;
    I *d_index_out;
    token_t *d_token_out;
    State<I>* d_index_states;
    State<state_t>* d_state_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_index_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_index_states, INDEX_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_state_states, STATE_STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, IN_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_index_out, INDEX_OUT_ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_token_out, TOKEN_OUT_ARRAY_BYTES));
    gpuAssert(cudaMemcpy(d_in, input, IN_ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    LexerCtx ctx = LexerCtx();
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(&ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states, size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size);
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
        lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(&ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states, size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_index_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    const I OUT_WRITE = temp_size * (sizeof(I) + sizeof(token_t));
    const I IN_READ = IN_ARRAY_BYTES;
    const I IN_STATE_MAP = sizeof(state_t) * size;
    const I SCAN_READ =  sizeof(state_t) * (size + size / 2); // Lowerbound, it does more work.
    compute_descriptors(temp, RUNS, IN_READ + IN_STATE_MAP + SCAN_READ + OUT_WRITE);
    free(temp);

    lexer<I, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(&ctx, d_in, d_index_out, d_token_out, d_state_states, d_index_states, size, NUM_LOGICAL_BLOCKS, d_dyn_index_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_index_out.data(), d_index_out, INDEX_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_token_out.data(), d_token_out, TOKEN_OUT_ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    /*
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Filter Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Filter Test Failed: Due to elements mismatch at index=" << i << std::endl;
            }
        } 
    }

    if (test_passes) {
        std::cout << "Filter test passed." << std::endl;
    }
    */
    

    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_token_out));
    gpuAssert(cudaFree(d_index_out));
    gpuAssert(cudaFree(d_index_states));
    gpuAssert(cudaFree(d_state_states));
    gpuAssert(cudaFree(d_dyn_index_ptr));
    gpuAssert(cudaFree(d_new_size));

    ctx.Cleanup();
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 2);
    size_t input_size;
    uint8_t* input = read_file(argv[1], &input_size);
    testLexer(input, input_size);

    free(input);
    gpuAssert(cudaPeekAtLastError());
    return 0;
}
