#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-35
#SBATCH --job-name=NPS_exp4_2
#SBATCH --mem=28GB
#SBATCH --constraint=8GB
#SBATCH -x node020,node023,node026,node021,node028,node094,node093,node098,node097,node094
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH --partition=use-everything
#SBATCH -D /om2/user/vanessad/understanding_reasoning/experiment_4

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES_
echo $CUDA_DEVICE_ORDER


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--offset_index 0 \
--output_folder results_NeurIPS_module_per_subtask_trial2 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--module_per_subtask True \
--run train

# --load_model True \
# # --module_per_subtask True \ this is for one module per subtask