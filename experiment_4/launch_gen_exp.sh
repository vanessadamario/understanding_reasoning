#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=baseline
#SBATCH --mem=1GB
#SBATCH -t 00:20:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om \
--experiment_index 0 \
--output_folder results_1 \
--root_data_folder /om/user/vanessad/understanding_reasoning/experiment_4/data_generation/datasets \
--run gen_exp
# --output_folder results_sqoop \
# --root_data_folder data_generation/sysgen_sqoop \
# --root_data_folder /om/user/vanessad/understanding_reasoning/experiment_4/data_generation/sysgen_sqoop \
