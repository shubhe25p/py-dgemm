. /opt/intel/oneapi/mkl/latest/env/vars.sh
. /opt/intel/oneapi/compiler/latest/env/vars.sh
cmake ../ -DBLA_VENDOR=Intel10_64ilp
make

# The Intel(R) Math Kernel Library (Intel(R) MKL) ILP64 
# libraries use the 64-bit integer type (necessary for indexing 
#large arrays, with more than 231-1 elements), 
# whereas the LP64 libraries index arrays with the 32-bit integer type.