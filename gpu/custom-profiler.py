#!/usr/bin/env python3

import time
import argparse
import math
import numpy as np

import cupy
xp=cupy

def initialize_accel_arrays( nsize, A, B):

    @cupy.fuse()
    def cupy_fuse_kernel(j, k):
        a = j*cupy.sin(j) + k*cupy.cos(k)
        b = k*cupy.cos(j) + j*cupy.sin(k)
        return a, b

    j, k= cupy.mgrid[0:nsize, 0:nsize]

    A[:], B[:] = cupy_fuse_kernel(j, k)
    cupy.cuda.runtime.deviceSynchronize()


#// -----
#// Function: create_arrays
#// allocate matrices and call their initialization functions
#// -----
def create_arrays(nsize, xp, devices ):

    def memory_string( memory_bytes ):
        units = ' kMGTPX'
        iunit = int( math.log( memory_bytes, 1024 ) )
        memory_units = memory_bytes / 1024**iunit
        memory_str = "{:.3f} {}B".format( memory_units, units[iunit] )
        return memory_str

    print("Preparing Matrix arrays")
    print("Memory required: {}".format( memory_string( 3 * nsize * nsize * 8 ) ) )

    print("Intializing Arrays in all devices")
    for i in devices:
        xp.cuda.runtime.setDevice(i)
        A = xp.zeros((nsize,nsize))
        B = xp.zeros((nsize,nsize))
        C = xp.zeros((nsize,nsize))
        initialize_accel_arrays( nsize, A, B )

    return A, B, C



        
#// ------------------------------------------------------- //
#// Function: matmul_loop
#//
#// Run & time matmul iterations.
#// This function should not be modified.
#// The call to xp.matmul should have the same signature, even if xp != Numpy
#// ------------------------------------------------------- //
def matmul_loop_async(niterations, A, B, C, xp, devices):


    print("Running matmul...")
    e1=[]
    e2=[]

    for i in devices:
        xp.cuda.runtime.setDevice(i)
        e1.append(xp.cuda.stream.Event())
        e2.append(xp.cuda.stream.Event())

    print("Warming up GPUs a bit")
    for i in range(10):
        xp.matmul(A,B,C)

    gpu_times=[[] for i in e1]

    for i in range(niterations):
        for e, device in zip(e1, devices):
            xp.cuda.runtime.setDevice(device)
            e.record()

        xp.matmul(A,B,C)

        for e, device in zip(e2,devices):
            xp.cuda.runtime.setDevice(device)
            e.record()

        for e, device in zip(e2,devices):
            xp.cuda.runtime.setDevice(device)
            e.synchronize()

        for i, (ev1, ev2) in enumerate(zip(e1, e2)):
            gpu_t = xp.cuda.get_elapsed_time(ev1, ev2) * 1e3
            gpu_times[i].append(gpu_t)

    return xp.asarray(gpu_times, dtype=np.float64)

    
#// Function: report_performance
def report_performance(nsize, gpu_times, devices, xp):
  

    flops = (2*nsize**3+ 2*nsize*nsize)  

    gflops = [[ flops / t / 1.0e3 for t in gpu_times[i]] for i in devices]

    print("Timing Report")
    for i in devices:
        print("GPU device ",i,"=",xp.mean(gpu_times[i]), "+-",xp.std(gpu_times[i]),"us", 
              "(min: ",xp.min(gpu_times[i])," max: ",xp.max(gpu_times[i]),")")
    
    print("GLOPS Report")
    for i in devices:
        print("GFLOPS device ",i,"=",xp.mean(xp.asarray(gflops[i])), "+-",xp.std(xp.asarray(gflops[i])),"GFLOPS",
                "(min: ",xp.min(xp.asarray(gflops[i])),"max: ",xp.max(xp.asarray(gflops[i])),")")

#// -----
#// Function: get_args
#// Parse and print the command line arguments
#// -----
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, required=False, default=10, help="number of iterations")
    parser.add_argument("--nsize", type=int, required=False, default=5004, help="dimension of square matrix")
    parser.add_argument("--device", type=int, required=False, default=0, help="device to use")
    parser.add_argument("--device-all", required=False, default=False, action='store_true', help="use all available devices")
    args = parser.parse_args()

    print("Requested Arguments:")
    print("  {:12s}: {}".format( "niterations", args.niterations ))
    print("  {:12s}: {}".format( "nsize",       args.nsize       ))
    if(not args.device_all):
        print("  {:12s}: {}".format( "device",      args.device      ))
    
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
    device      = args.device
    
    if args.device_all:
        devices = (0,1,2,3)
    else:
        devices = (device,)
    
    [ A, B, C ] = create_arrays( nsize, xp, devices )
    gpu_times = matmul_loop_async(niterations, A, B, C, xp, devices)
    report_performance(nsize, gpu_times, devices, xp)

if __name__ == '__main__':
    main()

