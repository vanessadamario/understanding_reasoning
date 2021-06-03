#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=150-179
#SBATCH --job-name=EXP3
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node003,node023,node026
#SBATCH -t 02:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.simg python main.py \
--host_filesystem om2 \
--modify_path True \
--output_path AWS_trial_2/ \
--root_data_folder path_folder/understanding_reasoning/experiment_3/data_generation/datasets \
--run test \
--offset_index 5000 \
--experiment_index ${SLURM_ARRAY_TASK_ID}