#pragma once
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda/mdspan>

static constexpr int BLOCK_SIZE = 128;

template<typename T>
using restrict_span = cuda::restrict_mdspan<T, cuda::std::dextents<std::size_t, 1>>;

template<typename T>
__global__ void kernel(restrict_span<const T> input, restrict_span<T> output)
{
    std::size_t n      = input.extent(0);
    std::size_t idx    = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * BLOCK_SIZE;

    for (std::size_t i = idx; i < n; i += stride) {
        T x = input[i];
        output[i] = T(1) / (T(1) + exp(-x));
    }
}

template<>
__global__ void kernel<__half>(restrict_span<const __half> input,
                               restrict_span<__half>       output)
{
    std::size_t n      = input.extent(0);
    std::size_t idx    = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * BLOCK_SIZE;

    for (std::size_t i = idx; i < n; i += stride) {
        float x = __half2float(input[i]);
        output[i] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

template __global__ void kernel<float>(restrict_span<const float>,
                                       restrict_span<float>);
template __global__ void kernel<double>(restrict_span<const double>,
                                        restrict_span<double>);

inline int compute_grid_size(std::size_t num_elements)
{
    int blocks = static_cast<int>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int max_blocks = 29184;
    return blocks < max_blocks ? blocks : max_blocks;
}
