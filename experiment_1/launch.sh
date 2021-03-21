#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=263
#SBATCH --job-name=query
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 12:00:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/experiment_1/output_slurm_query

module add openmind/singularity/3.4.1
hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om2 \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--load_model True \
--output_path **** here is your path to results \
--run train

# 1,2,7,8,9,27,28,29,36,43  train from scratch