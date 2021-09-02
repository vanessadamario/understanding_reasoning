#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=test
#SBATCH --array=0-35
#SBATCH --mem=20GB
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:1
#SBATCH -x node023,node020,node026,node021,node028,node094,node093,node098,node094,node023,node028,node097,dgx001
#SBATCH --constraint=8GB
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1
hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2 \
--data_path /om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_module_per_subtask_trial2 \
--run test \
--experiment_index ${SLURM_ARRAY_TASK_ID}


# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --data_path /om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets \
# --output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_revision \
# --run test

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --data_path /om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets \
# --output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_revision_trial2 \
# --run test

# --new_data_path True \
# --new_output_path True \