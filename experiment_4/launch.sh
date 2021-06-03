#!/bin/bash
#SBATCH -N 1
#SBATCH --array=
#SBATCH --job-name=trainEXP4
#SBATCH --mem=26GB
#SBATCH --constraint=8GB
#SBATCH -x node023,node026
#SBATCH --gres=gpu:1
#SBATCH -t 90:00:00
#SBATCH --partition=normal
#SBATCH -D path_to_folder/understanding_reasoning/experiment_4/slurm_output

module add clustername/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER


singularity exec -B /om2:/om2 --nv path_singularity/tensorflow2.5.0.simg python3 \
path_to_folder/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--offset_index 3600 \
--load_model True \
--output_folder spatial_only_second_trial \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run train


# spatial_only
# 181-200,201,202,203,204,205,206,207,208,209 without load
# 5-8,10,11,14,36,40,41,44,66-69,71,73,96-100,102-104,125-134,156-164,185-194,210-246,248,249
