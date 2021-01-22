#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=orig_sqoop
#SBATCH --array=0
#SBATCH --mem=20GB
#SBATCH -t 00:40:00
#SBATCH --gres=gpu:titan-x:1
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python main.py \
--host_filesystem om \
--output_folder results_sqoop_5 \
--experiment_index 2 \
--run test
