/*
 * bench.cu - nvbench harness for a 5-point heat equation stencil with
 * constant (Dirichlet) boundary conditions.  Only kernel.cuh is meant
 * to be modified for kernel logic.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "kernel.cuh"

static constexpr std::size_t DATA_BYTES = 256ULL * 1024 * 1024;

static bool running_under_ncu()
{
    return std::getenv("NV_COMPUTE_PROFILER_PERFWORKS_DIR") != nullptr;
}

template<typename T>
void stencil_cpu_ref(const T* u_in, T* u_out, int nx, int ny, T alpha)
{
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                u_out[j * nx + i] = u_in[j * nx + i];
                continue;
            }
            T center = u_in[j * nx + i];
            T left   = u_in[j * nx + (i - 1)];
            T right  = u_in[j * nx + (i + 1)];
            T up     = u_in[(j - 1) * nx + i];
            T down   = u_in[(j + 1) * nx + i];
            u_out[j * nx + i] = center + alpha * (left + right + up + down - T(4) * center);
        }
    }
}

using element_types = nvbench::type_list<float, double>;

template<typename T>
void kernel_bench(nvbench::state& state, nvbench::type_list<T>)
{
    const std::size_t total_elems = DATA_BYTES / sizeof(T);
    // Two NxN grids (input + output), so N = sqrt(total_elems / 2),
    // rounded down to a multiple of the block dimension.
    const int n  = static_cast<int>(std::sqrt(static_cast<double>(total_elems / 2)));
    const int nx = (n / BLOCK_DIM_X) * BLOCK_DIM_X;
    const int ny = nx;
    const T alpha = T(0.25);
    const std::size_t grid_elems = static_cast<std::size_t>(nx) * ny;

    thrust::host_vector<T> h_in(grid_elems, T(0));
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                h_in[j * nx + i] = T(1);

    thrust::device_vector<T> d_in  = h_in;
    thrust::device_vector<T> d_out(grid_elems, T(0));

    restrict_span_2d<const T> in_span(thrust::raw_pointer_cast(d_in.data()),
                                      ny, nx);
    restrict_span_2d<T>       out_span(thrust::raw_pointer_cast(d_out.data()),
                                       ny, nx);

    if (running_under_ncu()) {
        dim3 grid  = compute_grid_size(nx, ny);
        dim3 block = compute_block_size();
        kernel<T><<<grid, block>>>(in_span, out_span, alpha);
        cudaDeviceSynchronize();
        state.skip("Single-shot invocation under Nsight Compute");
        return;
    }

    state.add_element_count(static_cast<std::int64_t>(grid_elems), "NumElements");
    state.add_global_memory_reads<T>(grid_elems, "DataSize");
    state.add_global_memory_writes<T>(grid_elems, "DataSize");

    state.exec([=](nvbench::launch& launch) {
        dim3 grid  = compute_grid_size(nx, ny);
        dim3 block = compute_block_size();
        kernel<T><<<grid, block, 0, launch.get_stream()>>>(
            in_span, out_span, alpha);
    });

    thrust::host_vector<T> h_out(d_out);
    std::vector<T> h_ref(grid_elems);
    stencil_cpu_ref(h_in.data(), h_ref.data(), nx, ny, alpha);

    for (std::size_t idx = 0; idx < grid_elems; ++idx) {
        if (h_out[idx] != h_ref[idx])
            throw std::runtime_error("Correctness check failed for stencil");
    }
}

NVBENCH_BENCH_TYPES(kernel_bench, NVBENCH_TYPE_AXES(element_types))
    .set_type_axes_names({"T"});
