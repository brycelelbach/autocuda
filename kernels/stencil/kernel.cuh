#pragma once
#include <cstddef>
#include <cuda/mdspan>

static constexpr int BLOCK_DIM_X = 16;
static constexpr int BLOCK_DIM_Y = 16;

template<typename T>
using restrict_span_2d = cuda::restrict_mdspan<T, cuda::std::dextents<std::size_t, 2>>;

template<typename T>
__global__ void kernel(restrict_span_2d<const T> u_in,
                       restrict_span_2d<T>       u_out,
                       T alpha)
{
    int nx = static_cast<int>(u_in.extent(1));
    int ny = static_cast<int>(u_in.extent(0));
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        u_out(j, i) = u_in(j, i);
        return;
    }

    T center = u_in(j, i);
    T left   = u_in(j, i - 1);
    T right  = u_in(j, i + 1);
    T up     = u_in(j - 1, i);
    T down   = u_in(j + 1, i);

    u_out(j, i) = center + alpha * (left + right + up + down - T(4) * center);
}

template __global__ void kernel<float>(restrict_span_2d<const float>,
                                       restrict_span_2d<float>, float);
template __global__ void kernel<double>(restrict_span_2d<const double>,
                                        restrict_span_2d<double>, double);

inline dim3 compute_block_size()
{
    return dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
}

inline dim3 compute_grid_size(int nx, int ny)
{
    return dim3((nx + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (ny + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
}
