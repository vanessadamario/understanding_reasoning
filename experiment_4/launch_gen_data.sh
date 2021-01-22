#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=data
#SBATCH --mem=35GB
#SBATCH -t 00:25:00
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om \
--experiment_index 0 \
--dataset_name sqoop_variety_1 \
--variety 1 \
--run gen_data
