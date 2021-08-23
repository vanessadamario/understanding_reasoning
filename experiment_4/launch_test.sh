#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=test_NPS
#SBATCH --array=0-10
#SBATCH --mem=30GB
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node003,node007,node022,node020,node023,node026,node021,node028,node094,node093,node098,node097,node094
#SBATCH --partition=cbmm


module add openmind/singularity/3.4.1

hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp4 \
--output_folder sqoop_exps_0 \
--sqoop_dataset True \
--offset_index 0 \
--run test \
--experiment_index ${SLURM_ARRAY_TASK_ID}

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp4 \
--output_folder sqoop_exps_1 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 0 \
--sqoop_dataset True \
--run test

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp4 \
--output_folder sqoop_exps_2 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 0 \
--sqoop_dataset True \
--run test

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp4 \
--output_folder sqoop_exps_3 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 0 \
--sqoop_dataset True \
--run test

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
--host_filesystem om2_exp4 \
--output_folder sqoop_exps_4 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--offset_index 0 \
--sqoop_dataset True \
--run test


# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2_exp4 \
# --output_folder results_NeurIPS_module_per_subtask \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --offset_index 0 \
# --run test

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2_exp2 \
# --output_folder results_NeurIPS_module_per_subtask_trial2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --offset_index 0 \
# --run test

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python main.py \
# --host_filesystem om2_exp2 \
# --output_folder results_NeurIPS_revision_trial2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --offset_index 0 \
# --run test

#${SLURM_ARRAY_TASK_ID}
# --modify_path True \
# --root_data_folder /om/user/vanessad/understanding_reasoning/experiment_4/data_generation/datasets \