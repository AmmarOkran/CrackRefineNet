#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/shared/testing/output1.txt
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1
srun hostname
python3 test.py
srun sleep 100


module load /homes/ammarokran/miniconda3/