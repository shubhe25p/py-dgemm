. /opt/intel/oneapi/mkl/latest/env/vars.sh
. /opt/intel/oneapi/compiler/latest/env/vars.sh
cmake ../ -DBLA_VENDOR=Intel10_64ilp
make

# The Intel(R) Math Kernel Library (Intel(R) MKL) ILP64 
# libraries use the 64-bit integer type (necessary for indexing 
#large arrays, with more than 231-1 elements), 
# whereas the LP64 libraries index arrays with the 32-bit integer type.

# g++ benchmark.cpp -o benchmark -I$MKLROOT/include -L$MKLROOT/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core /opt/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libiomp5.so

