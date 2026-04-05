#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuComplex.h>

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
//            (src, dst, num_elements)
// where num_elements = 256 MiB / sizeof(T).
//
// Required explicit instantiations (below): int8_t, __half, float, double,
// cuDoubleComplex - add more if you extend bench.cu.
// =============================================================================
template<typename T>
__global__ void kernel(const T* __restrict__ src,
                       T* __restrict__       dst,
                       std::size_t           num_elements)
{
    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                               + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = tid; i < num_elements; i += stride) {
        dst[i] = src[i];
    }
}

template __global__ void kernel<int8_t>(const int8_t*, int8_t*, std::size_t);
template __global__ void kernel<__half>(const __half*, __half*, std::size_t);
template __global__ void kernel<float>(const float*, float*, std::size_t);
template __global__ void kernel<double>(const double*, double*, std::size_t);
template __global__ void kernel<cuDoubleComplex>(const cuDoubleComplex*,
                                                   cuDoubleComplex*,
                                                   std::size_t);

// Called by bench.cu to compute the grid dimension for <<<grid, BLOCK_SIZE>>>.
inline int compute_grid_size(std::size_t num_elements)
{
    return static_cast<int>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
