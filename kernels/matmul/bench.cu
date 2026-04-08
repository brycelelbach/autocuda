/*
 * bench.cu - nvbench harness for naive matrix multiplication C = A * B.
 * Only kernel.cuh is meant to be modified for kernel logic.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include <cuda_fp16.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "kernel.cuh"

NVBENCH_DECLARE_TYPE_STRINGS(__half, "F16", "half precision float");

static constexpr int MAT_DIM = 1024;

static bool running_under_ncu()
{
    return std::getenv("NV_COMPUTE_PROFILER_PERFWORKS_DIR") != nullptr;
}

template<typename T>
T make_val(float v) { return static_cast<T>(v); }

template<>
__half make_val<__half>(float v) { return __float2half(v); }

template<typename T>
double to_double(T v) { return static_cast<double>(v); }

template<>
double to_double<__half>(__half v) { return static_cast<double>(__half2float(v)); }

using element_types = nvbench::type_list<__half, float, double>;

template<typename T>
void kernel_bench(nvbench::state& state, nvbench::type_list<T>)
{
    const int M = MAT_DIM, N = MAT_DIM, K = MAT_DIM;
    const std::size_t elems_A = static_cast<std::size_t>(M) * K;
    const std::size_t elems_B = static_cast<std::size_t>(K) * N;
    const std::size_t elems_C = static_cast<std::size_t>(M) * N;

    T one = make_val<T>(1.0f);
    thrust::device_vector<T> d_A(elems_A, one);
    thrust::device_vector<T> d_B(elems_B, one);
    thrust::device_vector<T> d_C(elems_C);

    restrict_span_2d<const T> A_span(thrust::raw_pointer_cast(d_A.data()), M, K);
    restrict_span_2d<const T> B_span(thrust::raw_pointer_cast(d_B.data()), K, N);
    restrict_span_2d<T>       C_span(thrust::raw_pointer_cast(d_C.data()), M, N);

    if (running_under_ncu()) {
        dim3 grid  = compute_grid_size(M, N);
        dim3 block = compute_block_size();
        kernel<T><<<grid, block>>>(A_span, B_span, C_span);
        cudaDeviceSynchronize();
        state.skip("Single-shot invocation under Nsight Compute");
        return;
    }

    std::int64_t flops = 2LL * M * N * K;
    state.add_element_count(flops, "NumElements");
    state.add_global_memory_reads<T>(elems_A + elems_B, "DataSize");
    state.add_global_memory_writes<T>(elems_C, "DataSize");

    state.exec([=](nvbench::launch& launch) {
        dim3 grid  = compute_grid_size(M, N);
        dim3 block = compute_block_size();
        kernel<T><<<grid, block, 0, launch.get_stream()>>>(
            A_span, B_span, C_span);
    });

    // A = B = all-ones  =>  C[i][j] = K (exactly representable in all types)
    thrust::host_vector<T> h_C(d_C);
    double expected = static_cast<double>(K);
    for (std::size_t i = 0; i < h_C.size(); ++i) {
        if (std::abs(to_double(h_C[i]) - expected) > 0.5)
            throw std::runtime_error("Correctness check failed for matmul");
    }
}

NVBENCH_BENCH_TYPES(kernel_bench, NVBENCH_TYPE_AXES(element_types))
    .set_type_axes_names({"T"});
