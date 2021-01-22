#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=oos
#SBATCH --array=1542
#SBATCH --mem=20GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:titan-x:1
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python main.py \
--host_filesystem om \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--test_oos 1 \
--run test


# 335,364,375,387,399,478,484,495,507,520,412,423,434,447,459,537,542,556,567,579
# --test_oos 1 \