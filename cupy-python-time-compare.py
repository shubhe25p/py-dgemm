#!/usr/bin/env python3

import time
import argparse
import math
import numpy as np
##Hardware-specific python modules (including drop-in Numpy subtitutes) here
##substitute your own!
##NumPy substitutes should be aliased to "xp"
import cupy 
xp = cupy
from cupyx.profiler import benchmark
#// -----
#// Function: numpy_initializer
#// Switch between numpy(np) or accelerated-numpy(xp).
#// Device or library initialization may be added to this function if needed.
#// -----
def numpy_initializer( shownumpy ):

    global xp
    if accelerator:
        try:
            print("Using accelerated numpy:\n  {}\n".format( xp ) )
        except:
            print("The --accelerator option was used, but no accelerated numpy (e.g. cupy) was found.")
            exit(1)
    else:
        xp = np

    if shownumpy:
        print( xp.show_config() )

    return xp

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


#// ----
#// Function: copy_array_accel_to_host
#//This may be modified for non-cupy.
#// ----
def copy_array_accel_to_host( Ad, Ah ):
    Ah= Ad.get()


def matmul_loop(niterations, A, B, C, xp ):


    print("Running matmul...")

    tstart = time.time()
    synchronize_host_accel()
    tend = time.time()
    deltat = tend - tstart
    print("Synchronization Overhead (sec): {:.2e}".format( deltat ) )
       
    for i  in range(100):
        xp.matmul(A, B, out=C )
        synchronize_host_accel()
    
    deltat = np.zeros( niterations )
    for i in range(niterations):

        synchronize_host_accel()
        tstart = time.time()

        xp.matmul(A, B, out=C )

        synchronize_host_accel()
        tend = time.time()

        deltat[i] = tend - tstart

        if( i==0 ):
            print("First of {:d} iterations (sec): {:.6f}".format( niterations, deltat[0] ) )

    # sanity check array type
    #print("type of C:", type(C))

    return deltat

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

    t_start = time.time()
    A = xp.zeros((nsize,nsize))
    B = xp.zeros((nsize,nsize))
    C = xp.zeros((nsize,nsize))
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Allocation (sec): {:.6f}".format( deltat ) )
    

    t_start = time.time()
    if accelerator:
        # for i in range(4):
        #     xp.cuda.runtime.setDevice(i)
        initialize_accel_arrays( nsize, A, B )
    else:
        initialize_host_arrays( nsize, A, B )
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Initialization (sec): {:.3f}".format( deltat ) )

    return A, B, C

    
#// Function: report_performance
def report_performance(niterations, nsize, deltat_matmul ):
  

    flops = (2*nsize**3+ 2*nsize*nsize)  
    gflops = [ flops / t / 1.0e9 for t in deltat_matmul ]

    
    print("GPU FLOPS= {:7.2f}".format(xp.mean(np.asarray(gflops))))

#// -----
#// Function: get_args
#// Parse and print the command line arguments
#// -----
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, required=False, default=10, help="number of iterations")
    parser.add_argument("--nsize", type=int, required=False, default=5004, help="dimension of square matrix")
    parser.add_argument("--accelerator", required=False, default=True, action='store_true', help="option to use accelerator")
    parser.add_argument("--shownumpy", required=False, action='store_true', help="show numpy configuration")
    parser.add_argument("--testseed", type=int, required=False, default=None, help="random seed for sampling matrix elements to validate (integer).")
    parser.add_argument("--device", type=int, required=False, default=0, help="dimension of square matrix")

    args = parser.parse_args()

    print("Requested Arguments:")
    print("  {:12s}: {}".format( "niterations", args.niterations ))
    print("  {:12s}: {}".format( "nsize",       args.nsize       ))
    print("  {:12s}: {}".format( "accelerator", args.accelerator ))
    print("  {:12s}: {}".format( "testseed",    args.testseed    ))
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
    accelerator = args.accelerator
    niterations = args.niterations
    nsize       = args.nsize
    testseed    = args.testseed
    device      = args.device
    
    #choose the appropriate numpy-like interface:
    xp = numpy_initializer( args.shownumpy )
    gpu_times_python = []
    for i in range(4):
            xp.cuda.runtime.setDevice(i)
            [ A, B, C ] = create_arrays( nsize, xp )
            deltat_matmul = matmul_loop( niterations, A, B, C, xp )
            print("GPU",i,"time:",xp.mean(np.asarray(deltat_matmul)))

    # if correctness test has passed, report performance
            report_performance( niterations, nsize, deltat_matmul)
    print(benchmark(xp.matmul, (A, B, C), n_repeat=niterations, devices=(0,1,2,3)))
if __name__ == '__main__':
    main()

