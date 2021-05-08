#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=data10x
#SBATCH --array=0-209
# SBATCH --mem=30GB
#SBATCH -t 01:40:00
#SBATCH -x node023
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/10x_data/ \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--dataset_name dataset_31 \
--run convert


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/10x_data/ \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--dataset_name dataset_32 \
--run convert


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/10x_data/ \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--dataset_name dataset_33 \
--run convert

# --test_oos 1