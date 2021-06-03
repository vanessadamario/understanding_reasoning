#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=EXP2_10x
#SBATCH --array=600,602,603,605,608,667,698,728,850,851,852,853,854
#SBATCH --mem=30GB
#SBATCH -t 04:20:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node003,node023,node026,node020
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv path_to_singularity-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp2 \
--output_folder 10x_200k \
--modify_path True \
--root_data_folder path_to_folder/understanding_reasoning/experiment_2/data_generation/datasets \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 3000 \
--run test

