#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/sps.cu.h"
#include "../../common/util.cu.h"
#include "../../common/data.h"
#define PAD "%-38s "

template<typename T, typename I>
struct Tuple {
    T elem = T();
    I idx = I();
    bool flag = false;

    __host__ __device__ __forceinline__
    Tuple(T elem, I idx, bool flag) : elem(elem), idx(idx), flag(flag) {}
    
    __host__ __device__ __forceinline__
    Tuple() : elem(T()), idx(I()), flag(false) {}

    __host__ __device__ __forceinline__
    Tuple(const Tuple<T, I>& other) : elem(other.elem), idx(other.idx), flag(other.flag) {}

    __host__ __device__ __forceinline__
    Tuple(const volatile Tuple<T, I>& other) : elem(other.elem), idx(other.idx), flag(other.flag) {}
    
    __host__ __device__ __forceinline__
    Tuple<T, I>& operator=(const Tuple<T, I>& other) {
        if (this != &other) {
            elem = other.elem;
            idx = other.idx;
            flag = other.flag;
        }
        return *this;
    }

    __host__ __device__ __forceinline__
    Tuple<T, I>& operator=(const volatile Tuple<T, I>& other) volatile {
        if (this != &other) {
            elem = other.elem;
            idx = other.idx;
            flag = other.flag;
        }
        return *const_cast<Tuple<T, I>*>(this);
    }

    __host__ __device__ __forceinline__
    Tuple<T, I>& operator=(const Tuple<T, I>& other) volatile {
        if (this != &other) {
            elem = other.elem;
            idx = other.idx;
            flag = other.flag;
        }
        return *const_cast<Tuple<T, I>*>(this);
    }
};

template<typename T, typename I, typename OP>
struct AddTuple {
    OP op;

    __host__ __device__ __forceinline__
    AddTuple(OP op) : op(op) {}

    __device__ __forceinline__ Tuple<T, I> operator()(Tuple<T, I> a, Tuple<T, I> b) const {
        return Tuple<T, I>(
            b.flag ? b.elem : op(a.elem, b.elem),
            a.idx + b.idx,
            a.flag || b.flag
        );
    }
};


struct Add {
    __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) const {
        return a + b;
    }
};

template<typename T, typename I, typename OP, I BLOCK_SIZE, I ITEMS_PER_THREAD>
__global__ void
segreduce(T* d_in,
          bool* d_flags,
          T* d_out,
          volatile State<Tuple<T, I>>* states,
          I size,
          I num_logical_blocks,
          AddTuple<T, I, OP> op,
          volatile uint32_t* dyn_idx_ptr,
          volatile I* new_size) {
    volatile __shared__ Tuple<T, I> block[ITEMS_PER_THREAD * BLOCK_SIZE];
    volatile __shared__ Tuple<T, I> block_aux[BLOCK_SIZE];
    bool flags[ITEMS_PER_THREAD];

    uint32_t dyn_idx = dynamicIndex<uint32_t>(dyn_idx_ptr);
    I glb_offs = dyn_idx * BLOCK_SIZE * ITEMS_PER_THREAD;

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = i * blockDim.x + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size) {
            flags[i] = d_flags[(gid + 1) % size];
            block[lid] = Tuple<T, I>(d_in[gid], flags[i], d_flags[gid]);
        } else {
            flags[i] = false;
            block[lid] = Tuple<T, I>();
        }
    }

    __syncthreads();

    scan<Tuple<T, I>, I, AddTuple<T, I, OP>, ITEMS_PER_THREAD>(block, block_aux, states, op, Tuple<T, I>(), dyn_idx);

    #pragma unroll
    for (I i = 0; i < ITEMS_PER_THREAD; i++) {
        I lid = blockDim.x * i + threadIdx.x;
        I gid = glb_offs + lid;
        if (gid < size && flags[i]) {
            d_out[block[lid].idx - 1] = block[lid].elem;
        }
    }
    
    if (dyn_idx == num_logical_blocks - 1 && threadIdx.x == blockDim.x - 1) {
        *new_size = block[ITEMS_PER_THREAD * BLOCK_SIZE - 1].idx;
    }
    __syncthreads();
}

void testSegreduce(int32_t* vals, bool* flags, size_t _size, int32_t* expected, size_t expected_size) {
    using I = uint32_t;
    const I size = _size;
    const I BLOCK_SIZE = 256;
    const I ITEMS_PER_THREAD = 14;
    const I NUM_LOGICAL_BLOCKS = (size + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    const I ARRAY_BYTES = size * sizeof(int32_t);
    const I FLAG_ARRAY_BYTES = size * sizeof(bool);
    const I STATES_BYTES = NUM_LOGICAL_BLOCKS * sizeof(State<Tuple<int32_t, I>>);
    const I WARMUP_RUNS = 1000;
    const I RUNS = 10;

    std::vector<int32_t> h_out(size, 0);
    
    uint32_t* d_dyn_idx_ptr;
    I* d_new_size;
    int32_t *d_in, *d_out;
    bool *d_flags;
    State<Tuple<int32_t, I>>* d_states;
    gpuAssert(cudaMalloc((void**)&d_dyn_idx_ptr, sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**)&d_new_size, sizeof(I)));
    gpuAssert(cudaMalloc((void**)&d_flags, FLAG_ARRAY_BYTES));
    cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
    gpuAssert(cudaMalloc((void**)&d_states, STATES_BYTES));
    gpuAssert(cudaMalloc((void**)&d_in, ARRAY_BYTES));
    gpuAssert(cudaMalloc((void**)&d_out, ARRAY_BYTES));
    
    gpuAssert(cudaMemcpy(d_in, vals, ARRAY_BYTES, cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpy(d_flags, flags, FLAG_ARRAY_BYTES, cudaMemcpyHostToDevice));
    
    AddTuple<int32_t, I, Add> add = AddTuple<int32_t, I, Add>(Add());
    
    for (I i = 0; i < WARMUP_RUNS; ++i) {
        segreduce<int32_t, I, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_flags, d_out, d_states, size, NUM_LOGICAL_BLOCKS, add, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        gpuAssert(cudaPeekAtLastError());
    }

    timeval * temp = (timeval *) malloc(sizeof(timeval) * RUNS);
    timeval prev;
    timeval curr;
    timeval t_diff;

    for (I i = 0; i < RUNS; ++i) {
        gettimeofday(&prev, NULL);
        segreduce<int32_t, I, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_flags, d_out, d_states, size, NUM_LOGICAL_BLOCKS, add, d_dyn_idx_ptr, d_new_size);
        cudaDeviceSynchronize();
        gettimeofday(&curr, NULL);
        timeval_subtract(&t_diff, &curr, &prev);
        temp[i] = t_diff;
        cudaMemset(d_dyn_idx_ptr, 0, sizeof(uint32_t));
        cudaMemset(d_new_size, 0, sizeof(I));
        gpuAssert(cudaPeekAtLastError());
    }

    I temp_size = 0;
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));

    segreduce<int32_t, I, Add, BLOCK_SIZE, ITEMS_PER_THREAD><<<NUM_LOGICAL_BLOCKS, BLOCK_SIZE>>>(d_in, d_flags, d_out, d_states, size, NUM_LOGICAL_BLOCKS, add, d_dyn_idx_ptr, d_new_size);
    cudaDeviceSynchronize();
    gpuAssert(cudaMemcpy(h_out.data(), d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&temp_size, d_new_size, sizeof(I), cudaMemcpyDeviceToHost));
    
    bool test_passes = temp_size == expected_size;
    if (!test_passes) {
        std::cout << "Segreduce Test Failed: Expected size=" << expected_size << " but got size=" << temp_size << std::endl;
    } else {
        for (I i = 0; i < expected_size; ++i) {
            test_passes &= h_out[i] == expected[i];

            if (!test_passes) {
                std::cout << "Segreduce Test Failed: Due to elements mismatch at index=" << i << std::endl;
                break;
            }
        } 
    }

    if (test_passes) {
        compute_descriptors(temp, RUNS, FLAG_ARRAY_BYTES + ARRAY_BYTES + temp_size * sizeof(int32_t));
    }

    free(temp);
    gpuAssert(cudaFree(d_in));
    gpuAssert(cudaFree(d_out));
    gpuAssert(cudaFree(d_states));
    gpuAssert(cudaFree(d_dyn_idx_ptr));
    gpuAssert(cudaFree(d_new_size));
}

int main(int32_t argc, char *argv[]) {
    assert(argc == 3);
    int32_t* vals = NULL;
    size_t vals_size = 0;
    bool* flags = NULL;
    size_t flags_size = 0;
    read_i32_bool_array(argv[1], &vals, &vals_size, &flags, &flags_size);
    size_t expected_size;
    int32_t* expected = read_i32_array(argv[2], &expected_size);
    assert(vals_size == flags_size);
    /*
    int32_t a[6] = {1, 2, 3, 4, 5, 6};
    bool b[6] = {true, false, false, true, false, false};
    int32_t c[2] = {6, 15};
    testSegreduce(a, b, 6, c, 2);
    */
    printf("%s:\n", argv[1]);
    printf(PAD, "Segreduce:");
    testSegreduce(vals, flags, vals_size, expected, expected_size);
    free(vals);
    free(flags);
    free(expected);

    gpuAssert(cudaPeekAtLastError());
    return 0;
}
