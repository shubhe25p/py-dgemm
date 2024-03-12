#!/bin/bash

nsize=${1:-4096}
ml PrgEnv-gnu
ml python

export OMP_PLACES=cores
export OMP_PROC_BIND=true

numaSetting=0-15,128-143,112-127,240-255

for t in 16 32 64 128
do	
	output=$(OMP_NUM_THREADS=$t python python-dgemm.py --nsize $nsize --niterations 100)
	flops=$(echo $output | grep -oP "GLOPS AVG =\K.*")
	echo "OMP $t: $flops GLOPS"
done
