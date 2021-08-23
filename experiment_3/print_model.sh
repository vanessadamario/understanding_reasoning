#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=print
#SBATCH --mem=26GB
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1
hostname
singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
 python3 /om2/user/vanessad/understanding_reasoning/experiment_3/main.py \
 --host_filesystem om2 \
--output_path results_NeurIPS_revision \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run print
