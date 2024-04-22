#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=8
#SBATCH --time=10:00

module load python
srun -n 256 -c 2 python job-script-mpi.py