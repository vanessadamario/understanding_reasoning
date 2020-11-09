#!/bin/bash
#SBATCH -N 1
#SBATCH --array=12-15
#SBATCH -c 1
#SBATCH --job-name=baseline
#SBATCH --mem=20GB
#SBATCH -t 15:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run train