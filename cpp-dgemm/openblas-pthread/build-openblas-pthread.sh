# Comment this in if you want to disable OpenBLAS's CPU/Memory affinity handling.
# This feature is only implemented on Linux, and is always disabled on other platforms.
# Enabling affinity handling may improve performance, especially on NUMA systems, but 
# it may conflict with certain applications that also try to manage affinity.
# This conflict can result in threads of the application calling OpenBLAS ending up locked
# to the same core(s) as OpenBLAS, possibly binding all threads to a single core.
# For this reason, affinity handling is disabled by default. Can be safely enabled if nothing
# else modifies affinity settings.
# Note: enabling affinity has been known to cause problems with NumPy and R
# NO_AFFINITY = 1

# If you want to use the new, still somewhat experimental code that uses
# thread-local storage instead of a central memory buffer in memory.c
# Note that if your system uses GLIBC, it needs to have at least glibc 2.21
# for this to work.
# USE_TLS = 1

# When multithreading, OpenBLAS needs to use a memory buffer for communicating
# and collating results for individual subranges of the original matrix. Since
# the original GotoBLAS of the early 2000s, the default size of this buffer has
# been set at a value of 32<<20 (which is 32MB) on x86_64 , twice that on PPC.
# If you expect to handle large problem sizes (beyond about 30000x30000) uncomment
# this line and adjust the (32<<n) factor if necessary. Usually an insufficient value
# manifests itself as a crash in the relevant scal kernel (sscal_k, dscal_k etc) 
# BUFFERSIZE = 25

# The OpenMP scheduler to use - by default this is "static" and you
# will normally not want to change this unless you know that your main
# workload will involve tasks that have highly unbalanced running times
# for individual threads. Changing away from "static" may also adversely
# affect memory access locality in NUMA systems. Setting to "runtime" will
# allow you to select the scheduler from the environment variable OMP_SCHEDULE
# CCOMMON_OPT += -DOMP_SCHED=dynamic

make DYNAMIC_ARCH=0 BINARY=64 NO_LAPACK=0 TARGET=ZEN \
     NO_AFFINITY=0 USE_THREAD=1 NUM_THREADS=128 USE_OPENMP=0 \
     INTERFACE64=1 \
     
# About threaded BLAS. It will be automatically detected if you don't
# specify it.
# For force setting for single threaded, specify USE_THREAD = 0
# For force setting for multi  threaded, specify USE_THREAD = 1
# USE_THREAD = 0

# If you're going to use this library with OpenMP, please comment it in.
# This flag is always set for POWER8. Don't set USE_OPENMP = 0 if you're targeting POWER8.
# USE_OPENMP = 1

# Common Optimization Flag;
# The default -O2 is enough.
# Flags for POWER8 are defined in Makefile.power. Don't modify COMMON_OPT
# COMMON_OPT = -O2