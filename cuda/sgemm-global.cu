#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
const std::string errLogFile = "matrixValidationFailure.txt";

void randomize_matrix(float *matrix, int size) {
  for (int i = 0; i < size; i++) {
    matrix[i] = (float)rand() / RAND_MAX;
  }
}

__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {

    const uint BLOCKSIZE = 32;                            
   const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  gpuErrchk(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", 0, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
//   cublasHandle_t handle;
//   if (cublasCreate(&handle)) {
//     std::cerr << "Create cublas handle error." << std::endl;
//     exit(EXIT_FAILURE);
//   };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {4092, 8192};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 1.0, beta = 1.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  gpuErrchk(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  gpuErrchk(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  gpuErrchk(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  gpuErrchk(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  gpuErrchk(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    // if (kernel_num != 0) {
    //   run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
    //              handle); // cuBLAS
    //   run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
    //              handle); // Executes the kernel, modifies the result matrix
    //   gpuErrchk(cudaDeviceSynchronize());
    //   gpuErrchk(cudaGetLastError()); // Check for async errors during kernel run
    //   cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    //   cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    //   if (!verify_matrix(C_ref, C, m * n)) {
    //     std::cout
    //         << "Failed to pass the correctness verification against NVIDIA "
    //            "cuBLAS."
    //         << std::endl;
    //     if (m <= 128) {
    //       std::cout << " Logging faulty output into " << errLogFile << "\n";
    //       std::ofstream fs;
    //       fs.open(errLogFile);
    //       fs << "A:\n";
    //       print_matrix(A, m, n, fs);
    //       fs << "B:\n";
    //       print_matrix(B, m, n, fs);
    //       fs << "C:\n";
    //       print_matrix(C, m, n, fs);
    //       fs << "Should:\n";
    //       print_matrix(C_ref, m, n, fs);
    //     }
    //     exit(EXIT_FAILURE);
    //   }
    // }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_sgemm_coalesce(m, n, k, alpha, dA, dB, beta, dC);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    gpuErrchk(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);

  return 0;
};