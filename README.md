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

EARLY WORK STILL Work IN Progress