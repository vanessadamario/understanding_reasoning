#!/bin/bash
#SBATCH -N 1
#SBATCH --array=125,155
#SBATCH --job-name=sep_find_mod
#SBATCH --mem=30GB
#SBATCH --constraint=8GB
#SBATCH -x node023,node026
#SBATCH --gres=gpu:1
#SBATCH -t 10:30:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/experiment_4/slurm_output_shaping


module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--output_folder shaping \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run shaping


# 181-200,201,202,203,204,205,206,207,208,209 without loading
# 5-8,10,11,14,36,40,41,44,66-69,71,73,96-100,102-104,125-134,156-164,185-194,210-246,248,249
# 10x
# --load_model True \