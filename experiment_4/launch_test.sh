#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=EXP2_10x
#SBATCH --array=600,602,603,605,608,667,698,728,850,851,852,853,854
#SBATCH --mem=30GB
#SBATCH -t 04:20:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node003,node023,node026,node020
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp4 \
--output_folder sqoop_exps_0 \
--experiment_index 0 \
--offset_index 0 \
--run test

#${SLURM_ARRAY_TASK_ID}
# --modify_path True \
# --root_data_folder /om/user/vanessad/understanding_reasoning/experiment_4/data_generation/datasets \