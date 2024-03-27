#!/bin/bash

nsize=${1:-1000}
ml PrgEnv-nvidia
ml python

python python-dgemm.py --nsize $nsize --niterations 1000 --accelerator
