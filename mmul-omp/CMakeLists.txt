cmake_minimum_required(VERSION 3.16)
project(hw2 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE ON)

# find_package(BLAS REQUIRED) 

add_executable(benchmark-blas benchmark.cpp)
# target_link_libraries(benchmark-blas PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(benchmark-blas PRIVATE /global/homes/s/shubhp/work/py-dgemm/OpenBLAS/build/lib/libopenblas.so)
target_link_options(benchmark-blas PRIVATE -fopenmp)
target_compile_features(benchmark-blas PRIVATE c_std_11 c_restrict)
target_compile_options(benchmark-blas PRIVATE -Wall -pedantic -march=native)

# eof
