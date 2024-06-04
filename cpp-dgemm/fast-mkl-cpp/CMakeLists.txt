cmake_minimum_required(VERSION 3.16)
project(hw2 LANGUAGES C CXX)
set(PrgEnv-gnu)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(BLAS REQUIRED) # cray builds

add_executable(benchmark-blas benchmark.cpp)
target_link_libraries(benchmark-blas PRIVATE ${BLAS_LIBRARIES}) # cray builds
target_compile_options(benchmark-blas PRIVATE -Wall -pedantic -march=native)

# eof
