#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=190,191,220,223,227,228,229,230,231
#SBATCH --job-name=sep_FR
#SBATCH --mem=26GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 35:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
hostname
singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.simg python3 main.py \
--host_filesystem om2 \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run train


# #SBATCH --constraint=any-gpu
# --load_model 1 \