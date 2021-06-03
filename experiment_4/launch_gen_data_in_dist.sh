#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=spatial
#SBATCH --mem=80GB
#SBATCH -t 04:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 2 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_19 \
--spatial_only True \
--test_seen True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 5 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_20 \
--spatial_only True \
--test_seen True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 8 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_21 \
--spatial_only True \
--test_seen True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 10 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_22 \
--spatial_only True \
--test_seen True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 20 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_23 \
--spatial_only True \
--test_seen True \
--run gen_data


####


