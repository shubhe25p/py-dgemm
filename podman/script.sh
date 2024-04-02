#!/bin/bash

for t in 128
do
	for nsize in 16384
	do
		echo -e "OMP: $t with SIZE: $nsize \n"
		OMPSTR="OMP_NUM_THREADS=$t"
		#echo $OMPSTR
		podman-hpc run --env $OMPSTR --env "OMP_PLACES=cores" --env "OMP_PROC_BIND=true" n10_pydgemm:1.0 python /opt/py-dgemm/python-dgemm.py --nsize $nsize --niteration 100 | grep Best
	done
done


