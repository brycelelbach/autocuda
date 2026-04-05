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

add_executable(bench bench.cu)
target_link_libraries(bench PRIVATE nvbench::main)
# kernel.cuh lives next to bench.cu
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
