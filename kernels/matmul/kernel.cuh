#pragma once
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda/mdspan>

static constexpr int BLOCK_DIM = 16;

template<typename T>
using restrict_span_2d = cuda::restrict_mdspan<T, cuda::std::dextents<std::size_t, 2>>;

template<typename T>
__global__ void kernel(restrict_span_2d<const T> A,
                       restrict_span_2d<const T> B,
                       restrict_span_2d<T>       C)
{
    int M = static_cast<int>(A.extent(0));
    int K = static_cast<int>(A.extent(1));
    int N = static_cast<int>(B.extent(1));

    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (row >= M || col >= N) return;

    T sum = T(0);
    for (int k = 0; k < K; ++k)
        sum += A(row, k) * B(k, col);

    C(row, col) = sum;
}

template<>
__global__ void kernel<__half>(restrict_span_2d<const __half> A,
                               restrict_span_2d<const __half> B,
                               restrict_span_2d<__half>       C)
{
    int M = static_cast<int>(A.extent(0));
    int K = static_cast<int>(A.extent(1));
    int N = static_cast<int>(B.extent(1));

    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += __half2float(A(row, k)) * __half2float(B(k, col));

    C(row, col) = __float2half(sum);
}

template __global__ void kernel<float>(restrict_span_2d<const float>,
                                       restrict_span_2d<const float>,
                                       restrict_span_2d<float>);
template __global__ void kernel<double>(restrict_span_2d<const double>,
                                        restrict_span_2d<const double>,
                                        restrict_span_2d<double>);

inline dim3 compute_block_size()
{
    return dim3(BLOCK_DIM, BLOCK_DIM);
}

inline dim3 compute_grid_size(int M, int N)
{
    return dim3((N + BLOCK_DIM - 1) / BLOCK_DIM,
                (M + BLOCK_DIM - 1) / BLOCK_DIM);
}
