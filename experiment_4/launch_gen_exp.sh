#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=baseline
#SBATCH --mem=1GB
#SBATCH -t 00:20:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om2_exp4 \
--experiment_index 0 \
--output_folder results_NeurIPS_module_per_subtask_trial2 \
--root_data_folder /om2/user/vanessad/understanding_reasoning/experiment_4/data_generation/datasets \
--run gen_exp