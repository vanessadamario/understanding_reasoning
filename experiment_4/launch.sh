#!/bin/bash
#SBATCH -N 1
#SBATCH --array=3
#SBATCH -c 1
#SBATCH --job-name=sqoop2
#SBATCH --mem=6GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 12:00:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python main.py \
--host_filesystem om \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--output_folder results_sqoop_2 \
--load_model 1 \
--run train

# --load_model 1 \