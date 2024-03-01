import cupy as cp
from cupyx.profiler import benchmark

def f(a, b):
    return 3 * cp.sin(-a) * b

a = 0.5 - cp.random.random((100,))
b = cp.random.random((100,))
print(benchmark(f, (a, b), n_repeat=1000, devices=(0,1,2,3)))
