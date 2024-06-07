#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J dgemm
#SBATCH --mail-user=spachchigar@sfsu.edu
#SBATCH --mail-type=ALL
#SBATCH -A nintern
#SBATCH -t 0:45:0

ml PrgEnv-gnu
ml python

conda activate test-openmp

export OMP_PLACES=cores
export OMP_PROC_BIND=true

# numaSetting=0-15,128-143,112-127,240-255
declare -a ompsched=("static" "dynamic" "guided")

# for t in 16 32 64 128
# do	
# 	output=$(OMP_NUM_THREADS=$t python python-dgemm.py --nsize $nsize --niterations 100)
# 	flops=$(echo $output | grep -oP "GLOPS AVG =\K.*")
# 	echo "OMP $t: $flops GLOPS"
# done

count=10

for i in $(seq 1 $count)
do
	echo "Iteration $i"
	for sched in "${ompsched[@]}"
	do
		for chunk in 1 2 4 8 16 32 64 128 256 512 1024
		do
			echo "OMP $sched $chunk"
			output=$(OMP_SCHEDULE=$sched,$chunk OMP_NUM_THREADS=128 OMP_DISPLAY_ENV=true python python-dgemm.py --nsize 4096)
			output=$(echo $output | grep -oP "Best \K.*")
			echo $output
			echo "Time left: $(squeue -h -j $SLURM_JOBID -o %L)"
		done
	done
done
