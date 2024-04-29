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


## Tips and Tricks

### Goal:

The goal is to get maximum performance on DGEMM like operations in Python. 

### Five BLAS libraries tested:

- Intel MKL 
- Cray LibSci
- OpenBLAS
- BLIS
- NetLib

### General guidelines for multithreaded code

- Use OMP_PROC_BIND=true and OMP_PLACES=cores to achieve optimal performance. Other options get similar result.

### Intel MKL

- Perlmutter runs on AMD EPYC 7763 double socket chips, Intel MKL is highly optimized for its own hardware, although a quick view of linux profiler do 
show that MKL uses a zen optimized kernel it almost never uses any AVX2 instructions.

- NERSC has developed a module with a workaround where it uses a Intel Haswell kernel with AVX2 instructions, expect a performance gain of about 20% with this.

```
module load fast-mkl-amd
```

### Cray LibSci

- Expect the best performance out of this BLAS library, because its running on a cray system.

- Cray LibSci linux profiling shows that it actually runs Naples kernel a first gen system while a Milan (current system) is third gen, as of this writing configuring LIBSCI_ARCH_OVERRIDE for cray libsci still switch to Naples kernel.

### OpenBLAS

- OpenBLAS builds with two variants if built from conda-forge: pthreads and openmp

-  The default openblas version is a "pthread" variant which is not affected by OMP_NUM_THREADS and OpenMP variant does gets affected, with this new variant expect a 4x speedup

- OpenBLAS ZEN3 kernel uses Intel Haswell codes with some optimizations for Zen 3.

- Best practices from OpenBLAS: OpenMP provides its own locking mechanisms, so when your code makes BLAS/LAPACK calls from inside OpenMP parallel regions it is imperative that you use an OpenBLAS that is built with USE_OPENMP=1, as otherwise deadlocks might occur. Furthermore, OpenBLAS will automatically restrict itself to using only a single thread when called from an OpenMP parallel region. When it is certain that calls will only occur from the main thread of your program (i.e. outside of omp parallel constructs), a standard pthreads build of OpenBLAS can be used as well. In that case it may be useful to tune the linger behaviour of idle threads in both your OpenMP program (e.g. set OMP_WAIT_POLICY=passive) and OpenBLAS (by redefining the THREAD_TIMEOUT variable at build time, or setting the environment variable OPENBLAS_THREAD_TIMEOUT smaller than the default 26) so that the two alternating thread pools do not unnecessarily hog the cpu during the handover.

- conda-forge sets the max thread limit to 128 for OpenBLAS build.

Build Pthread variant

```
conda create -n testblas -c conda-forge -y python=3.11 numpy "libblas=*=*openblas"
```

Build OpenMP variant

```
conda create -n testblas -c conda-forge -y python=3.11 numpy "libblas=*=*openblas" "libopenblas=*=openmp*"
```

### BLIS

- Suddenly BLIS stops working for OMP_PLACES=cores OMP_PROC_BIND=true still don't know why, it seems for this setting only a single thread run

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
