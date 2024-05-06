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

## Tips and Tricks

### Goal:

The goal is to get maximum performance on DGEMM like operations in Python. 

### Five BLAS libraries tested:

- Intel MKL 
- Cray LibSci
- OpenBLAS
- BLIS
- NetLib


### Intel MKL

- Perlmutter runs on AMD EPYC 7763 double socket chips, Intel MKL is highly optimized for its own hardware, although a quick view of linux profiler do 
show that MKL uses a zen optimized kernel it almost never uses any AVX2 instructions.

- NERSC has developed a module with a workaround where it uses a Intel Haswell kernel with AVX2 instructions, expect a performance gain of about 20% with this.

- Strong scaling study reveals that at a range of of about 512-1024 of matrix size, expect a performance boost after which it drop for a short time while again increasing for larger matrices, similar effect in fast-mkl-amd module.

- The reason for the momentary drop increase then drop in performance might be due to some architecture difference between Intel and AMD chips

- PROC BIND=true and PLACES=cores leads to a max performance, other options like OMP_PROC_BIND=spread yield similar results.

```
module load fast-mkl-amd
```

### Cray LibSci

- Expect the best performance out of this BLAS library, because its running on a cray system.

- Cray LibSci linux profiling shows that it actually runs Naples kernel a first gen system while a Milan (current system) is third gen, as of this writing configuring LIBSCI_ARCH_OVERRIDE for cray libsci still switch to Naples kernel.

- Ideal performance gains in strong as well as weak scaling study.

- Cray gives about 2x more GFLOPs then any BLAS implementations

- 

### OpenBLAS

- OpenBLAS builds with two variants if built from conda-forge: pthreads and openmp

-  The default openblas version is a "pthread" variant which is not affected by OMP_NUM_THREADS and OpenMP variant does gets affected, with this new variant expect a 4x speedup

- OpenBLAS ZEN3 kernel uses Intel Haswell codes with some optimizations for Zen 3.

- Be careful when requesting openmp threads inside a BLAS routine or vice-versa, BLAS threads can interfere with OMP threads and can lead to resource contention.
  
- conda-forge sets the max thread limit to 128 for OpenBLAS build.

Build Pthread variant

```
conda create -n testblas -c conda-forge -y python=3.11 numpy "libblas=*=*openblas"
```

Build OpenMP variant

```
conda create -n testblas -c conda-forge -y python=3.11 numpy "libblas=*=*openblas" "libopenblas=*=openmp*"
```

- Default Pthread variant gives mediocre GFLOPS till 8192 then a sudden boost after that, unknown reasons again, most run gives half GFLOPs then Intel MKL.
- OpenMP build variant on the other hand gives comparable performance to MKL but still slow compared to Cray Libsci

### BLIS

- BLIS is really strange, it gives worst performance for PROC_BIND=true and PLACES=cores where it only runs on single physical core while for PROC_BIND=false and PLACES=cores it gives the best performance. Here again BLIS can be built with both pthread and openmp variant but for some unknown reason conda-forge provides only the pthreads built.

- More investigation needed!!

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
