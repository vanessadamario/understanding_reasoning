#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=gen_exps
#SBATCH --mem=2GB
#SBATCH -t 00:20:00
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3 main.py \
--host_filesystem om2 \
--experiment_index 0 \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_revision/ \
--run gen_exp
