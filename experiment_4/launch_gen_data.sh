#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=spatial
#SBATCH --mem=80GB
#SBATCH -t 04:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 1 \
--root_data_folder /om2/user/vanessa/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_18 \
--spatial_only True \
--h5_file True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 2 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_31 \
--spatial_only True \
--h5_file True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 5 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_32 \
--spatial_only True \
--h5_file True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 8 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_33 \
--spatial_only True \
--h5_file True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 10 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_34 \
--spatial_only True \
--h5_file True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity/tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--variety 20 \
--root_data_folder path_to_folder/understanding_reasoning/experiment_4/data_generation/datasets/ \
--dataset_name dataset_35 \
--spatial_only True \
--h5_file True \
--run gen_data