#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=query_test
#SBATCH --array=1-104
#SBATCH --mem=20GB
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH -x node023,node026
#SBATCH --constraint=8GB
#SBATCH --partition=normal

module add cluster/singularity/3.4.1

singularity exec -B /om2:/om2 --nv path_to_singularity_tensorflow2.5.0.simg python main.py \
--host_filesystem om2 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--new_data_path True \
--new_output_path True \
--data_path path_to_folder/understanding_reasoning/experiment_1/data_generation/datasets \
--output_path path_to_folder/understanding_reasoning/experiment_1/10x_data \
--run test