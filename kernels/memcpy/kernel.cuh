#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cuda/mdspan>

// =============================================================================
// TUNABLE PARAMETERS - the only section you should modify
// =============================================================================

// Thread-block width (must be a multiple of 32).
static constexpr int BLOCK_SIZE = 256;

// =============================================================================
// Kernel - copies num_elements values of type T from src to dst.
//
// bench.cu registers one nvbench benchmark with a type axis over T; it launches:
//   kernel<T><<<compute_grid_size(num_elements), BLOCK_SIZE, 0, stream>>>
//            (src, dst)
// where num_elements = 256 MiB / sizeof(T).
//
// Required explicit instantiations (below): int8_t, __half, float, double,
// cuDoubleComplex - add more if you extend bench.cu.
// =============================================================================

template<typename T>
using restrict_span = cuda::restrict_mdspan<T, cuda::std::dextents<std::size_t, 1>>;

template<typename T>
__global__ void kernel(restrict_span<const T> src,
                       restrict_span<T>       dst)
{
    const std::size_t num_elements = src.extent(0);
    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                               + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = tid; i < num_elements; i += stride) {
        dst[i] = src[i];
    }
}

template __global__ void kernel<int8_t>(restrict_span<const int8_t>,
                                        restrict_span<int8_t>);
template __global__ void kernel<__half>(restrict_span<const __half>,
                                        restrict_span<__half>);
template __global__ void kernel<float>(restrict_span<const float>,
                                       restrict_span<float>);
template __global__ void kernel<double>(restrict_span<const double>,
                                        restrict_span<double>);
template __global__ void kernel<cuDoubleComplex>(restrict_span<const cuDoubleComplex>,
                                                  restrict_span<cuDoubleComplex>);

// Called by bench.cu to compute the grid dimension for <<<grid, BLOCK_SIZE>>>.
inline int compute_grid_size(std::size_t num_elements)
{
    return static_cast<int>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
