cmake_minimum_required(VERSION 3.30.4)

project(autocuda LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Disable nvbench's own tests / examples / install rules when used as a
# subdirectory - they are not needed here.
set(NVBench_ENABLE_TESTING       OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_EXAMPLES      OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_INSTALL_RULES OFF CACHE BOOL "" FORCE)

add_subdirectory(/home/blelbach/dev/nvbench nvbench-build)

function(add_kernel_bench name)
    set(dir ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${name})
    add_executable(bench_${name} ${dir}/bench.cu)
    target_link_libraries(bench_${name} PRIVATE nvbench::main)
    target_include_directories(bench_${name} PRIVATE ${dir})
endfunction()

add_kernel_bench(memcpy)
add_kernel_bench(stencil)
add_kernel_bench(matmul)
add_kernel_bench(sigmoid)
