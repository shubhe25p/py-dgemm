# NERSC 10 Python DGEMM Compute Benchmark

This benchmark uses matrix multiplication 
to measure a processor's sustainable rate of
double-precision floating point operations.
A secondary goal is to demonstrate good
performance when using Python language;
python-dgemm uses the NumPy library
as an interface to the underlying `DGEMM` function.
Drop-in replacements for NumPy may be used
to run the benchmark on GPUs or other accelerators.

(c) 2024 Shubh Pachchigar
Performance Engineering Intern
NERSC, Berkeley Lab

## Tips and Tricks (TODO: add plots, Work in Progress)

### Goal:

The goal is to get maximum performance on DGEMM like operations in Python for CPUs. 

### Five BLAS libraries tested:

- Intel MKL 
- Cray LibSci
- OpenBLAS
- BLIS
- AOCC-BLAS
- NetLib


### Intel MKL in Numpy

- Perlmutter runs on AMD EPYC 7763 double socket chips, Intel MKL is highly optimized for its own hardware, although a quick view of linux profiler do 
show that MKL uses a zen optimized kernel it almost never uses any AVX2 instructions.

- NERSC has developed a module with a workaround where it uses a Intel Haswell kernel with AVX2 instructions, expect a performance gain of about 20% with this.

- Strong scaling study reveals that at a range of of about 512-1024 of matrix size, expect a performance boost after which it drop for a short time while again increasing for larger matrices, similar effect in fast-mkl-amd module.

- The reason for the momentary drop increase then drop in performance might be due to some architecture difference between Intel and AMD chips

- PROC BIND=true and PLACES=cores leads to a max performance, other options like OMP_PROC_BIND=spread yield similar results. Reason for better performance is that when threads are moved between cores, cache invalidation happens on the original cores so we do not benefit from cache hierarchy. If your application is using caches a lot like DGEMM here then it is best to set these flags.

- Most of the time multithreading in a BLAS library is enabed by either Pthreads/OpenMP and yes both are not the same. Intel developed independently an OpenMP implementation(libiomp.so) which shares its roots from the LLVM OpenMP implementation(libomp.so). 

- A quick linux profiling shows that numpy uses libomp.so for multithreading and thus it accepts all the LLVM env variables as well as OpenMP environment variables. If the code is compiled with GNU then LLVM OpenMP might have empty symbols for some GNU OpenMP functions.

- At the end, Intel MKL is good has some strange scaling behavior at size 512 and size 1024(OMP_NUM_THREADS=128, OMP_PROC_BIND=true, OMP_PLACES=cores). Also number of iterations matter as well as if the number of iterations increases for some very strange reason size 1024 reaches a really high peak not sure why though, but generally for the benchmark included here for 10 iterations, GFLOPs increases till 512 then drops(and no L2/L3 cache size does not play a role here) then increasing again and cause expected performance.


### Tricking MKL

- Although MKL has a dgemm kernel for Zen it never uses AVX2 instructions, but in a similar architecture Intel Haswell it does so people have found a clever way to create a shared object file and LD_PRELOAD it before running the actual dgemm. The amd-mkl folder has the shared object file and there is only a single symbol inside that returns true when MKL checks if it is a Intel CPU or not. 

- A module is created out of this trick at NERSC which essentially I think just sets the LD_PRELOAD env value to this shared object file and voila code runs faster. 

- And the flaws of MKL and its strange behavior is also seen here

```
module load fast-mkl-amd
```

### Cray LibSci

- Expect the best performance out of this BLAS library, because its running on a cray system.

- Cray LibSci linux profiling shows that it actually runs Naples kernel a first gen system while a Milan (current system) is third gen, as of this writing configuring LIBSCI_ARCH_OVERRIDE for cray libsci still switch to Naples kernel.

- Ideal performance gains in strong as well as weak scaling study.

- Cray gives about 2x more GFLOPs then any BLAS implementations

- In different programming environment(PrgEnv-cray and PrgEnv-gnu) this behaves slightly differently, the OpenMP runtime library differs, in case of Cray PE it symlinks to libcraymp.so and in case of GNU PE it symlinks to libgomp.so

- Do not expect a performance difference between these systems and again only OpenMP version is provided here.

### OpenBLAS

- OpenBLAS builds with two variants if built from conda-forge: pthreads and openmp

-  The default openblas version is a "pthread" variant which is not affected by OMP_NUM_THREADS and OpenMP variant does gets affected, with this new variant expect a 4x speedup

- OpenBLAS ZEN3 kernel uses Intel Haswell codes with some optimizations for Zen 3.

- Be careful when requesting openmp threads inside a BLAS routine or vice-versa, BLAS threads can interfere with OMP threads and can lead to resource contention.
  
- conda-forge sets the max thread limit to 128 for OpenBLAS build.

#### Build Pthread variant

- Conda forge behaves in a very weird way when openblas-pthreads is install it gets libgomp(GNU OpenMP) and when OpenMP build is requested it installs libomp(LLVM OpenMP). One of the reasons might that libgomp is not fork safe. But I do not understand why install an OpenMP implementation in a pthreads build where it is not expected to receive or use any OpenMP settings. This version is significantly slower than OpenMP build, please use openblas_openmp.

```
conda create -n testblas -c conda-forge -y python=3.11 numpy "libblas=*=*openblas"
```

#### Build OpenMP variant

- This is much faster than Pthreads, uses LLVM OpenMP.
```
conda create -n testblas -c conda-forge -y python=3.11 numpy "libblas=*=*openblas" "libopenblas=*=openmp*"
```

- Default Pthread variant gives mediocre GFLOPS till 8192 then a sudden boost after that, unknown reasons again, most run gives half GFLOPs then Intel MKL.
- OpenMP build variant on the other hand gives comparable performance to MKL but still slow compared to Cray Libsci

### BLIS

- BLIS exhibits intriguing behavior, with its worst performance observed when configured with PROC_BIND=true and PLACES=cores/threads, running solely on a single physical core with an atomic barrier. However, it performs optimally under PROC_BIND=false and PLACES=cores settings. Despite BLIS source offering both Pthread and OpenMP threading model, conda-forge provides only the pthreads version.

- Despite exhaustive efforts, including consideration of BLIS versions and profiling attempts, diagnosing the issue has proven challenging due to the exceedingly slow execution speed, yielding a mere 1 GFLOP.

- Several factors contribute to this puzzle. Upon creating a fresh environment, it first installs the pthreads variant of BLIS which might be due to its superior performance over the OpenMP version. 

- Conda's installation of libgomp introduces additional complexity, interfering with pthreads when OMP_PROC_BIND/OMP_PLACES are set. Although technically, setting the number of OMP_NUM_THREADS is not necessary in a pthreads build, but there is a way to manipulate threads in BLIS_Pthreads with BLIS_NUM_THREADS. However, activation of libgomp results in competition for compute resources between Pthreads and OpenMP causing a sort of resource deadlock(more investigation needed), as evidenced by profiling data indicating prolonged periods spent in atomic barriers. Notably, my pthreads build responds to OMP environment variables, indicating both Pthreads and OpenMP are being used in BLIS.

- Removing libgomp from the Conda environment resolves the conflict, automatically replacing it with LLVM OpenMP. However, this change results in BLIS no longer being affected by any OMP environment variables.

- While it's clear that libgomp is not at fault, there appears to be an issue with how numpy interacts with it.

Specify a build number and version number for BLAS
```
conda create -n test-blis -c conda-forge numpy python=3.11.8 "libblas=*=*blis" "blis=0.9.0=hd590300_1
```


### NetLib

- Do not expect any performance gains, extremely slow


### Pyomp at NERSC

In bare metal containers, pyomp for current nersc-python module does not work, to make it work:

```
conda create -n pyomp python=3.9
conda install Python-for-HPC::numba Python-for-HPC::llvmlite -c conda-forge 
```
