#!/bin/bash

nsize=${1:-1000}
ml PrgEnv-gnu
ml python

export OMP_PLACES=cores
export OMP_PROC_BIND=true

numaSetting=0-15,128-143

for t in 32
do	
	output=$(OMP_NUM_THREADS=$t numactl -C $numaSetting python python-dgemm.py --nsize $nsize)
	flops=$(echo $output | grep -oP "GLOPS AVG =\K.*")
	echo "OMP $t: $flops GLOPS"
done
