#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=data
#SBATCH --mem=40GB
#SBATCH -t 00:25:00
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om2 \
--experiment_index 0 \
--offset_index 0 \
--train_combinations 2 \
--n_train_per_question 10000 \
--dataset_name dataset_0 \
--run gen_data



# --n_train_per_question 10000 \
# --positive_train_combinations 1 \
# --negative_train_combinations 1 \
# --positive_test_combinations 1 \
# --negative_test_combinations 1  \