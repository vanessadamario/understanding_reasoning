#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=1-159
#SBATCH --job-name=modulation
#SBATCH --mem=13GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node023,node026
#SBATCH -t 90:00:00
#SBATCH --partition=normal
#SBATCH -D path_to_folder/understanding_reasoning/experiment_1/output_slurm_pilot

module add clustername/singularity/3.4.1
hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv path_to_singularity \
python path_to_folder/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--output_path path_to_folder/understanding_reasoning/experiment_1/stem_modulation \
--run train

#tensorflow2.5.0.simg
# 1,2,7,8,9,27,28
# ,29,36,43  train from scratch