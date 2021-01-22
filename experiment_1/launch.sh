#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=175
#SBATCH --job-name=complete_
#SBATCH --mem=8GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 10:00:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1
hostname
singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om2 \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--load_model 1 \
--run train

# final retrain
# 21,121,123,125,137,139,141,143,147,149,151,153,155,157,159,161,167,169,171,175,177,179,183,203,209

# --load_model 1 \
#SBATCH --constraint=any-gpu

#