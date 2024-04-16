from numba import njit
from numba.openmp import openmp_context as openmp

@njit
def pi_numba_openmp(num_steps):
    step = 1.0/num_steps
    sum = 0.0
    with openmp("parallel for private(x) reduction(+:sum)"):
        for i in range(num_steps):
            x = (i+0.5)*step
            sum += 4.0/(1.0 + x*x)
    pi = step*sum
    return pi

pi = pi_numba_openmp(1_000_000_000)
print(pi)