#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=gen_exps
#SBATCH --mem=2GB
#SBATCH -t 00:20:00
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2 \
--output_path results_NeurIPS_revision_trial2/ \
--experiment_index 0 \
--run gen_exp