//
// (C) 2021, E. Wes Bethel
// benchmark-* hardness for running different versions of matrix multiply
//    over different problem sizes
//
// usage: [-N problemSizeInt] [-B blockSizeInt]
// 
// On the command line you may optionally set the problem size (-N problemSizeInt),
// as well as optionally set the block size (-B blockSizeInt).
//
// If you specify nothing on the command line, the benchmark will iterate through a 
// prescribed set of problem sizes, which are defined in the code below.
//
// For the blocked version, if you don't specify a block size on the command line,
// then the benchmark will iterate of a prescribed set of block sizes, which are
// defined in the code below.
/*
make LIBNAMESUFFIX=omp USE_OPENMP=1 USE_THREAD=1 NUM_THREADS=128 NO_LAPACK=1 NO_AFFINITY=1 
*/
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

#include <cblas.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

void fill(double* p, int n) {
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}

void report_performance(double* arr, int iterations, int size){

   std::cout << " First of " << iterations << " iterations= " << arr[0] <<" (sec)" << std::endl;
   double flops = (double)(2*size*size*size + 2*size*size);
   std::vector<double> gflops;
   double max=0.0;
   int index=0;
   for(int i=0;i<iterations;i++){
      gflops.push_back(flops/(arr[i]*1000000));
      if(gflops[i] > max){
         max = gflops[i];
         index=i;
      }
   }
   std::cout << "Best (" << index << ")      " << arr[index] << " sec     " << gflops[index] << " GFLOPs" << std::endl;

}

/* The benchmarking program */
int main(int argc, char** argv) 
{
   std::cout << "Description:\t" << std::endl << std::endl;

   // check to see if there is anything on the command line:
   // -N nnnn    to define the problem size
   // -B bbbb    to define the block size
   int cmdline_N = -1;
   int cmdline_I = -1; 
   int c;

   while ( (c = getopt(argc, argv, "N:I:")) != -1) {
      switch(c) {
         case 'N':
            cmdline_N = std::atoi(optarg == NULL ? "-999" : optarg);
            // std::cout << "Command line problem size: " << cmdline_N << std::endl;
            break;
         case 'I':
            cmdline_I = std::atoi(optarg == NULL ? "10" : optarg);
            break;
      }
   }


   std::cout << std::fixed << std::setprecision(6);


   // set up the problem sizes
   int default_problem_sizes[] = {2048};
   std::vector<int> test_sizes;
   std::vector<double> exec_time;

   if (cmdline_N > 0)
      test_sizes.push_back(cmdline_N);
   else
   {
      for (int i : default_problem_sizes)
         test_sizes.push_back(i);
   }


   /* For each test size */
   for (int i=0;i<cmdline_I;i++){

   for (int n : test_sizes) 
   {
      printf("Working on problem size N=%d \n", n);

         // allocate memory for 6 NxN matrics
         std::vector<double> buf(6 * n * n);
         double* A = buf.data() + 0;
         double* B = A + n * n;
         double* C = B + n * n;
         double* Acopy = C + n * n;
         double* Bcopy = Acopy + n * n;
         double* Ccopy = Bcopy + n * n;

         // load up matrics with some random numbers
         fill(A, n * n);
         fill(B, n * n);
         fill(C, n * n);

         // make copies of A, B, C for use in verification of results
         memcpy((void *)Acopy, (const void *)A, sizeof(double)*n*n);
         memcpy((void *)Bcopy, (const void *)B, sizeof(double)*n*n);
         memcpy((void *)Ccopy, (const void *)C, sizeof(double)*n*n);

         // insert timer code here
         std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

         cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 1., C, n);

         std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

         std::chrono::duration<double> elapsed = end_time - start_time;

         std::cout << " Elapsed time is : " << elapsed.count() << " (sec) " << std::endl;
         exec_time.push_back(elapsed.count());

   } // end loop over problem sizes
   }
   report_performance(exec_time.data(), cmdline_I, cmdline_N);
   return 0;
}

// EOF

// EOF
