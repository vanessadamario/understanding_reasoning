#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=test
#SBATCH --array=3,4
#SBATCH --mem=20GB
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:1
#SBATCH -x node020,node022,node023,node026
#SBATCH --constraint=8GB
#SBATCH --partition=use-everything

module add openmind/singularity/3.4.1
hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --data_path /om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets \
# --output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_revision \
# --run test

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--data_path /om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_revision_trial2 \
--run test

# --new_data_path True \
# --new_output_path True \