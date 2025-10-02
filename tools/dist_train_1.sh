#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/homes/ammarokran/output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1
source ~/.bashrc
source activate openmmlab
python /homes/ammarokran/mmsegmentation/configs/crack_last/linknet_efficientnet_b5_cracks_AdamW_0.0001_0.0005dec_5000warm.py