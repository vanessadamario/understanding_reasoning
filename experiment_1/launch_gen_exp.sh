#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=gen_exps
#SBATCH --mem=2GB
#SBATCH -t 00:20:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

singularity exec -B /om2:/om2 --nv path_singularity python main.py \
--host_filesystem om2 \
--experiment_index 0 \
--output_path path_to_folder/understanding_reasoning/experiment_1/10x_data/ \
--run gen_exp