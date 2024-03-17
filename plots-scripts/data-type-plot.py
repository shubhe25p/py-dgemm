# python cupy-data-type-study.py --nsize 4096 --niterations 100 --accelerator, warmup 100 iterations

GFLOPS_INT8 = 13515.41
GFLOPS_INT16 = 11915.63
GFLOPS_INT32 = 13190.62
GFLOPS_INT64 = 3526.02
GFLOPS_FLOAT16 = 222433.68
GFLOPS_FLOAT32 = 18811.11
GFLOPS_FLOAT64 = 18927.44

import matplotlib.pyplot as plt

# Creating a 10x10 array
data = [[GFLOPS_INT8, GFLOPS_INT16, GFLOPS_INT32, GFLOPS_INT64, GFLOPS_FLOAT16, GFLOPS_FLOAT32, GFLOPS_FLOAT64]]

# create a line plot

plt.plot(data[0], marker='o', color='b')
plt.title( "GFLOPS Comparison for different data types A100" )
plt.xlabel('Data Types')
plt.ylabel('GFLOPS')
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['INT8', 'INT16', 'INT32', 'INT64', 'FLOAT16', 'FLOAT32', 'FLOAT64'])
plt.show()