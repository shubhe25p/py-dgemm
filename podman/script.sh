#!/bin/bash

for t in 64 128 256 512 1024 2048
do
	for nsize in 1024 2048 4096 8192 16384
	do
		echo -e "OMP: $t with SIZE: $nsize \n"
		OMPSTR="OMP_NUM_THREADS=$t"
		#echo $OMPSTR
		podman-hpc run --env $OMPSTR n10_pydgemm:1.0 python /opt/py-dgemm/python-dgemm.py --nsize $nsize --niteration 100 | grep Best
	done
done


