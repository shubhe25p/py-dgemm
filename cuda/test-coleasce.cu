#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>


__global__ void test_coleasce(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  printf("Block DIM X: %d, Block DIM Y: %d, Block IDX X: %d, Block IDX Y: %d, Thread IDX X: %d, Thread IDX Y: %d, X: %d, Y: %d\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y);
}

void run_test_coleasce(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  
  test_coleasce<<<1, 8>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }

  printf("Running kernel %d on device %d.\n", 0, deviceIdx);
  run_test_coleasce(m, n, k, alpha, dA, dB, beta, dC);
  return 0;
};