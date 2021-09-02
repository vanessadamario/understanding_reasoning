#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0,13
#SBATCH --job-name=EXP3
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node023,node020,node026,node021,node028,node094,node093,node098,node094,node023,node028,node097,dgx001
#SBATCH -t 02:00:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1
hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2 \
--output_path results_NeurIPS_module_per_subtask_trial2/ \
--run test \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID}


# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2 \
# --output_path results_NeurIPS_revision/ \
# --run test \
# --offset_index 0 \
# --experiment_index ${SLURM_ARRAY_TASK_ID}

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2 \
# --output_path results_NeurIPS_revision_trial2/ \
# --run test \
# --offset_index 0 \
# --experiment_index ${SLURM_ARRAY_TASK_ID}


# singularity exec -B /om2:/om2 --nv path_singularity_tensorflow2.simg python main.py \
# --host_filesystem om2 \
# --modify_path True \
# --output_path AWS_trial_2/ \
# --root_data_folder path_folder/understanding_reasoning/experiment_3/data_generation/datasets \
# --run test \
# --offset_index 5000 \
# --experiment_index ${SLURM_ARRAY_TASK_ID}