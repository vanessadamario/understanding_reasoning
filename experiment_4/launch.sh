#!/bin/bash
#SBATCH -N 1
#SBATCH --array=7,8,9,37,38,39,66,67,68,69,95,96,97,98,99,125,126,127,128,129,155,156,157,159,185,186,189,210,211,212,213,214,220,221,222,223,224,230,231,232,233,242,244,10,11,12,13,42,44,73,74,100,101,103,104,130,131,160,161,162,163,164,192,193,194,215,216,217,218,219,225,226,227,228,229,235,238,239,247,248,249
#SBATCH --job-name=10x_26
#SBATCH --mem=5GB
#SBATCH --constraint=8GB
#SBATCH -x node023,node026
#SBATCH --gres=gpu:1
#SBATCH -t 40:00:00
#SBATCH --partition=normal
#SBATCH -D /om2/user/vanessad/understanding_reasoning/experiment_2/slurm_output


module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp2 \
--output_folder 10x_200k \
--load_model True \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run train
