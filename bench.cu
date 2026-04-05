/*
 * bench.cu - fixed nvbench harness (multi-element-type via type axis).
 *
 * One benchmark, `kernel_bench`, sweeps T over int8, fp16, fp32, fp64, and
 * complex fp64. Only kernel.cuh is meant to be modified for kernel logic.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "kernel.cuh"

inline bool operator==(const cuDoubleComplex& a, const cuDoubleComplex& b)
{
    return cuCreal(a) == cuCreal(b) && cuCimag(a) == cuCimag(b);
}

// 256 MiB total traffic per type - element count scales with sizeof(T).
static constexpr std::size_t DATA_BYTES = 256ULL * 1024 * 1024;

static bool running_under_ncu()
{
    return std::getenv("NV_COMPUTE_PROFILER_PERFWORKS_DIR") != nullptr;
}

template<typename T>
struct bench_fill;

template<>
struct bench_fill<int8_t>
{
    static int8_t src() { return 1; }
    static int8_t dst() { return 0; }
};

template<>
struct bench_fill<__half>
{
    static __half src() { return __float2half(1.f); }
    static __half dst() { return __float2half(0.f); }
};

template<>
struct bench_fill<float>
{
    static float src() { return 1.f; }
    static float dst() { return 0.f; }
};

template<>
struct bench_fill<double>
{
    static double src() { return 1.0; }
    static double dst() { return 0.0; }
};

template<>
struct bench_fill<cuDoubleComplex>
{
    static cuDoubleComplex src() { return make_cuDoubleComplex(1.0, 0.0); }
    static cuDoubleComplex dst() { return make_cuDoubleComplex(0.0, 0.0); }
};

template<typename T>
void run_kernel_bench(nvbench::state& state, T init_src, T init_dst)
{
    const std::size_t num_elements = DATA_BYTES / sizeof(T);
    thrust::device_vector<T> src(num_elements, init_src);
    thrust::device_vector<T> dst(num_elements, init_dst);

    restrict_span<const T> src_span(thrust::raw_pointer_cast(src.data()), num_elements);
    restrict_span<T>       dst_span(thrust::raw_pointer_cast(dst.data()), num_elements);

    if (running_under_ncu()) {
        const int grid = compute_grid_size(num_elements);
        kernel<T><<<grid, BLOCK_SIZE>>>(src_span, dst_span);
        cudaDeviceSynchronize();
        state.skip("Single-shot invocation under Nsight Compute");
        return;
    }

    state.add_element_count(num_elements, "NumElements");
    state.add_global_memory_reads<T>(num_elements, "DataSize");
    state.add_global_memory_writes<T>(num_elements, "DataSize");

    state.exec([src_span, dst_span, num_elements](nvbench::launch& launch) {
        const int grid = compute_grid_size(num_elements);
        kernel<T><<<grid, BLOCK_SIZE, 0, launch.get_stream()>>>(
            src_span, dst_span);
    });

    thrust::host_vector<T> h_src(src);
    thrust::host_vector<T> h_dst(dst);
    if (!std::equal(h_src.begin(), h_src.end(), h_dst.begin()))
        throw std::runtime_error("Correctness check failed: dst != src");
}

using element_types = nvbench::type_list<int8_t, __half, float, double, cuDoubleComplex>;

template<typename T>
void kernel_bench(nvbench::state& state, nvbench::type_list<T>)
{
    run_kernel_bench(state, bench_fill<T>::src(), bench_fill<T>::dst());
}

NVBENCH_BENCH_TYPES(kernel_bench, NVBENCH_TYPE_AXES(element_types))
    .set_type_axes_names({"T"});
