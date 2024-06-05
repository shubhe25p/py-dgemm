cmake_minimum_required(VERSION 3.16)
project(hw2 LANGUAGES C CXX)
set(PrgEnv-gnu)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE ON)

add_executable(benchmark-blas benchmark.cpp)
target_link_libraries(benchmark-blas PRIVATE $ENV{HOME}/work/py-dgemm/cpp-dgemm/aocc-blas-omp/blis/lib/zen3/libblis-mt.so)
target_compile_options(benchmark-blas PRIVATE -Wall -pedantic -march=native)

# eof
