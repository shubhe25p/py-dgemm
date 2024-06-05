# BLIS disables multithreading by default. In order to allow multithreaded parallelism from BLIS, you must first enable multithreading explicitly at configure-time.

# As of this writing, BLIS optionally supports multithreading via OpenMP or POSIX threads(or both)

./configure --enable-threading=openmp --enable-cblas \
            --prefix=$HOME/work/py-dgemm/cpp-dgemm/blis-openmp/lib \
            --enable-verbose-make CC=gcc zen3\
            
            