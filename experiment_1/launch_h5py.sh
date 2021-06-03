#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=data10x
#SBATCH --array=0-209
# SBATCH --mem=30GB
#SBATCH -t 01:40:00
#SBATCH -x node023
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
path_to_folder/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--output_path path_to_folder/understanding_reasoning/experiment_1/10x_data/ \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--dataset_name dataset_31 \
--test_seen True \
--run convert


singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
path_to_folder/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--output_path path_to_folder/understanding_reasoning/experiment_1/10x_data/ \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--dataset_name dataset_32 \
--test_seen True \
--run convert


singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
path_to_folder/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--output_path path_to_folder/understanding_reasoning/experiment_1/10x_data/ \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--dataset_name dataset_33 \
--test_seen True \
--run convert

# --test_oos 1