/*
 * bench.cu - nvbench harness for a PyTorch-style sigmoid operator:
 *   output[i] = 1 / (1 + exp(-input[i]))
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

// 2 GiB per buffer, 4 GiB total footprint (input + output) per type.
static constexpr std::size_t DATA_BYTES = 2ULL * 1024 * 1024 * 1024;

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
    const std::size_t num_elements = DATA_BYTES / sizeof(T);

    T zero = make_val<T>(0.0f);
    thrust::device_vector<T> d_input(num_elements, zero);
    thrust::device_vector<T> d_output(num_elements);

    restrict_span<const T> in_span(thrust::raw_pointer_cast(d_input.data()),
                                   num_elements);
    restrict_span<T>       out_span(thrust::raw_pointer_cast(d_output.data()),
                                    num_elements);

    if (running_under_ncu()) {
        const int grid = compute_grid_size(num_elements);
        kernel<T><<<grid, BLOCK_SIZE>>>(in_span, out_span);
        cudaDeviceSynchronize();
        state.skip("Single-shot invocation under Nsight Compute");
        return;
    }

    state.add_element_count(static_cast<std::int64_t>(num_elements), "NumElements");
    state.add_global_memory_reads<T>(num_elements, "DataSize");
    state.add_global_memory_writes<T>(num_elements, "DataSize");

    state.exec([=](nvbench::launch& launch) {
        const int grid = compute_grid_size(num_elements);
        kernel<T><<<grid, BLOCK_SIZE, 0, launch.get_stream()>>>(
            in_span, out_span);
    });

    // sigmoid(0) = 0.5 exactly
    thrust::host_vector<T> h_output(d_output);
    for (std::size_t i = 0; i < num_elements; ++i) {
        if (std::abs(to_double(h_output[i]) - 0.5) > 1e-5)
            throw std::runtime_error("Correctness check failed for sigmoid");
    }
}

NVBENCH_BENCH_TYPES(kernel_bench, NVBENCH_TYPE_AXES(element_types))
    .set_type_axes_names({"T"});
