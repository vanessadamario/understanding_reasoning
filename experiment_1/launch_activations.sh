#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=ACT_half-sep_find
#SBATCH --array=0
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node003,node023,node026,node022
#SBATCH -t 01:00:00
#SBATCH --partition=normal

module add cluster/singularity/3.4.1

hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 --dataset_name dataset_15 --architecture_type half-sep_find \
--output_path path_to_folder/understanding_reasoning/experiment_1/results_AWS \
--new_output_path True \
--new_data_path True \
--experiment_case 1 \
--run activations

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 --dataset_name dataset_16 --architecture_type half-sep_find \
--output_path path_to_folder/understanding_reasoning/experiment_1/results_AWS \
--new_output_path True \
--new_data_path True \
--experiment_case 1 \
--run activations

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 --dataset_name dataset_17 --architecture_type half-sep_find \
--output_path path_to_folder/understanding_reasoning/experiment_1/results_AWS \
--new_output_path True \
--new_data_path True \
--experiment_case 1 \
--run activations

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 --dataset_name dataset_18 --architecture_type half-sep_find \
--output_path path_to_folder/understanding_reasoning/experiment_1/results_AWS \
--new_output_path True \
--new_data_path True \
--experiment_case 1 \
--run activations

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 --dataset_name dataset_26 --architecture_type half-sep_find \
--output_path path_to_folder/understanding_reasoning/experiment_1/results_AWS \
--new_output_path True \
--new_data_path True \
--experiment_case 1 \
--run activations

singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 --dataset_name dataset_27 --architecture_type half-sep_find \
--output_path path_to_folder/understanding_reasoning/experiment_1/results_AWS \
--new_output_path True \
--new_data_path True \
--experiment_case 1 \
--run activations

# singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
# --host_filesystem om2 \
# --experiment_index 0 --dataset_name dataset_1 --architecture_type  \
# --output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping \
# --run activations


# singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
# --host_filesystem om2 \
# --experiment_index 0 --dataset_name dataset_18 --architecture_type sep_res \
# --output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping \
# --run activations


# singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
# --host_filesystem om2 \
# --experiment_index 0 --dataset_name dataset_19 --architecture_type sep_res \
# --output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping \
# --run activations


# singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.5.0 python3 main.py \
# --host_filesystem om2 \
# --experiment_index 0 --dataset_name dataset_17 --architecture_type sep_res \
# --output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping \
# --run activations

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 main.py \
# --host_filesystem om2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --output_path path_to_folder/understanding_reasoning/experiment_1/fixstar_results/results/ \
# --new_output_path True \
# --new_data_path True \
# --data_path path_to_folder/understanding_reasoning/experiment_1/data_generation/datasets \
# --run test

# xboix-tensorflow-latest-tqm.simg
# --test_oos 1 \
# --on_validation True \
# #SBATCH --gres=gpu:1
#SBATCH --constraint=2G
#