#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=data
#SBATCH --mem=40GB
#SBATCH -t 02:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

# singularity exec -B /om2:/om2 --nv path_to_singularity_2.5.0 python3 main.py \
# --host_filesystem om2 \
# --experiment_index 0 \
# --dataset_name dataset_30 \
# --h5_file True \
# --output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping \
# --n_train_per_question 100000 \
# --positive_train_combinations 1 \
# --negative_train_combinations 1 \
# --positive_test_combinations 5 \
# --negative_test_combinations 5 \
# --run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity_2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 \
--dataset_name dataset_31 \
--h5_file True \
--output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping/ \
--n_train_per_question 100000 \
--positive_train_combinations 2 \
--negative_train_combinations 2 \
--positive_test_combinations 5 \
--negative_test_combinations 5 \
--test_seen True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity_2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 \
--dataset_name dataset_32 \
--h5_file True \
--output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping/ \
--n_train_per_question 100000 \
--positive_train_combinations 5 \
--negative_train_combinations 5 \
--positive_test_combinations 5 \
--negative_test_combinations 5 \
--test_seen True \
--run gen_data

singularity exec -B /om2:/om2 --nv path_to_singularity_2.5.0 python3 main.py \
--host_filesystem om2 \
--experiment_index 0 \
--dataset_name dataset_33 \
--h5_file True \
--output_path path_to_folder/understanding_reasoning/experiment_1/query_early_stopping/ \
--n_train_per_question 100000 \
--positive_train_combinations 8 \
--negative_train_combinations 8 \
--positive_test_combinations 5 \
--negative_test_combinations 5 \
--test_seen True \
--run gen_data

# --n_train_per_question 10000 \
# --positive_train_combinations 1 \
# --negative_train_combinations 1 \
# --positive_test_combinations 1 \
# --negative_test_combinations 1  \