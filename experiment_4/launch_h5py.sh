#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=evalEXP4
#SBATCH --array=121-124,126-129,131-134,136-139,141-144,146-149,151-154,156-159,161-164,166-169,171-174,176-179,181-184,186-189,191-194,196-199,201-204,206-209,282-323
#SBATCH --mem=30GB
#SBATCH -t 00:40:00
#SBATCH -x node023
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH --partition=normal
#SBATCH -D path_to_folder/understanding_reasoning/experiment_4/slurm_output

module add openmind/singularity/3.4.1

singularity exec -B /om2:/om2 --nv path_to_singularity-tensorflow-latest-tqm.simg python3 \
path_to_folder/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--on_validation True \
--run test

singularity exec -B /om2:/om2 --nv path_to_singularity-tensorflow-latest-tqm.simg python3 \
path_to_folder/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run test

singularity exec -B /om2:/om2 --nv path_to_singularity-tensorflow-latest-tqm.simg python3 \
path_to_folder/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--test_seen True \
--run test

# --test_oos 1