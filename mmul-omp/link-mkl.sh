. /opt/intel/oneapi/mkl/latest/env/vars.sh
. /opt/intel/oneapi/compiler/latest/env/vars.sh
cmake ../ -DBLA_VENDOR=Intel10_64lp
make
