#!/bin/env bash

#SBATCH -o job1.out				  # name of output file for this submission script
#SBATCH -e joberr.err				  # name of error file for this submission script

#SBATCH -c 24

julia -p 24 cluster_job.jl 2>&1