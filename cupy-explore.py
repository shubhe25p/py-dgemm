#!/usr/bin/env python3

import time
import argparse
import math
import numpy as np

import cupy 
xp = cupy
from cupyx.profiler import benchmark


#// -----
#// Function: synchronize_host_accel
#// This is a no-op if running on the host
#// May be modified for non-cupy
#// -----
def synchronize_host_accel():
    if accelerator:
        cupy.cuda.runtime.deviceSynchronize()
        

#// -----
#// Function: initialize_accel_arrays
#// Initialize matrices using accelerator memory/processor
#// Here, cupy arrays are used to run this custom kernel on 
#// May be modified for non-cupy
#// -----
def initialize_accel_arrays( nsize, A, B ):

    @cupy.fuse()
    def cupy_fuse_kernel(j, k):
        a = j*cupy.sin(j) + k*cupy.cos(k)
        b = k*cupy.cos(j) + j*cupy.sin(k)
        return a, b

    j, k = cupy.mgrid[0:nsize, 0:nsize]
    A[:], B[:] = cupy_fuse_kernel(j, k)
    cupy.cuda.runtime.deviceSynchronize()

    

#// -----
#// CODE BELOW THIS LINE SHOULD NOT BE MODIFIED
#// -----

#// -----
#// Function: initialize_host_arrays
#// Initialize matrices using host memory/processor
#// -----
def initialize_host_arrays( nsize, A, B ):
    for j in range(nsize):
        for k in range(nsize):
            A[j, k] = j*math.sin(j) + k*math.cos(k)
            B[j, k] = k*math.cos(j) + j*math.sin(k)


#// -----
#// Function: create_arrays
#// allocate matrices and call their initialization functions
#// -----
def create_arrays(nsize, xp ):

    def memory_string( memory_bytes ):
        units = ' kMGTPX'
        iunit = int( math.log( memory_bytes, 1024 ) )
        memory_units = memory_bytes / 1024**iunit
        memory_str = "{:.3f} {}B".format( memory_units, units[iunit] )
        return memory_str

    print("Preparing Matrix arrays")
    print("Memory required: {}".format( memory_string( 3 * nsize * nsize * 8 ) ) )
    #xp.cuda.runtime.setDevice(1)
    t_start = time.time()
    A = xp.zeros((nsize,nsize))
    B = xp.zeros((nsize,nsize))
    C = xp.zeros((nsize,nsize))
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Allocation (sec): {:.6f}".format( deltat ) )
    

    t_start = time.time()
    initialize_accel_arrays( nsize, A, B )
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Initialization (sec): {:.3f}".format( deltat ) )
    print("CUDA DEVICE=",xp.cuda.get_device_id())
    return A, B, C



        
#// ------------------------------------------------------- //
#// Function: matmul_loop
#//
#// Run & time matmul iterations.
#// This function should not be modified.
#// The call to xp.matmul should have the same signature, even if xp != Numpy
#// ------------------------------------------------------- //
def matmul_loop(niterations, A, B, C, xp, devices):


    print("Running matmul...")
    e1=[]
    e2=[]

    for i in devices:
        xp.cuda.runtime.setDevice(i)
        e1.append(xp.cuda.Event())
        e2.append(xp.cuda.Event())

    xp.cuda.runtime.setDevice(0)
    print("Warming up GPU a bit")
    for i in range(10):
        xp.matmul(A,B,C)

    for e, device in zip(e1, devices):
        xp.cuda.runtime.setDevice(device)
        e.record()
        # e.synchronize()

    gpu_times=[[] for i in e1]
    for i in range(niterations):
        for e, device in zip(e1, devices):
            xp.cuda.runtime.setDevice(device)
            e.record()

        xp.matmul(xp.asarray(A),xp.asarray(B),xp.asarray(C))

        for e, device in zip(e2,devices):
            xp.cuda.runtime.setDevice(device)
            e.record()

        for e, device in zip(e2,devices):
            xp.cuda.runtime.setDevice(device)
            e.synchronize()

        for i, (ev1, ev2) in enumerate(zip(e1, e2)):
            gpu_t = xp.cuda.get_elapsed_time(ev1, ev2) * 1e3
            gpu_times[i].append(gpu_t)

    print("CUDA DEVICE=",xp.cuda.get_device_id())
    return np.asarray(gpu_times, dtype=np.float64)

    
#// Function: report_performance
def report_performance(niterations, nsize, deltat_matmul ):
  

    flops = (2*nsize**3+ 2*nsize*nsize)  
    gflops = [ flops / t / 1.0e9 for t in deltat_matmul ]

    print("GLOPS AVG NORMAL TIME= {:7.2f}".format(xp.mean(np.asarray(gflops))))
    #xp.set_printoptions(precision=2)
    print(gflops)

#// -----
#// Function: get_args
#// Parse and print the command line arguments
#// -----
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, required=False, default=10, help="number of iterations")
    parser.add_argument("--nsize", type=int, required=False, default=5004, help="dimension of square matrix")
    args = parser.parse_args()

    print("Requested Arguments:")
    print("  {:12s}: {}".format( "niterations", args.niterations ))
    print("  {:12s}: {}".format( "nsize",       args.nsize       ))
    
    return args

#// ------------------------------------------------------- //
#// Function: main
#// -----
def main():

    #retreive command line arguments
    args = get_args()

    #stores accelerator as a global variable
    global accelerator
    accelerator = True
    niterations = args.niterations
    nsize       = args.nsize

    #choose the appropriate numpy-like interface:
    [ A, B, C ] = create_arrays( nsize, xp )
    gpu_times = matmul_loop( niterations, A, B, C, xp, devices=(0,1,2,3))
    for i in range(4):
        print("GPU",i,"=",xp.mean(gpu_times[i]))
    


    # if correctness test has passed, report performance
    #report_performance( niterations, nsize, gpu_times)
    #print(benchmark(xp.matmul, (A, B, C), n_repeat=niterations, devices=(0,1,2,3)))
if __name__ == '__main__':
    main()

