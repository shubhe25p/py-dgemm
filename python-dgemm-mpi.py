from mpi4py import MPI
import numpy as np
import math
# Get the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the global matrix size
global_rows = 8000
global_cols = 8000

# Divide the work among processes
local_rows = global_rows // size
if rank < global_rows % size:
    local_rows += 1
local_cols = global_cols

# Create the local matrix
A_local = np.zeros((local_rows, local_cols), dtype=np.float64)
B_local = np.zeros((local_rows, local_cols), dtype=np.float64)
C_local = np.zeros((local_rows, local_cols), dtype=np.float64)

# Initialize the local matrix
for j in range(local_rows):
    for k in range(local_cols):
        A_local[j, k] = j*math.sin(j) + k*math.cos(k)
        B_local[j, k] = k*math.cos(j) + j*math.sin(k)

# Gather the local matrices into the global matrix
A = None
B = None
C = None
if rank == 0:
    C = np.zeros((global_rows, global_cols), dtype=np.float64)
    B = np.zeros((global_rows, global_cols), dtype=np.float64)
    A = np.zeros((global_rows, global_cols), dtype=np.float64)


comm.Gatherv(A_local.flatten(), [A, (local_rows * local_cols, None), MPI.DOUBLE], root=0)
comm.Gatherv(B_local.flatten(), [B, (local_rows * local_cols, None), MPI.DOUBLE], root=0)

if rank == 0:
    print(A)