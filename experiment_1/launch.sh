#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=18-53
#SBATCH --job-name=NPS_exp1_1
#SBATCH --mem=28GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node023,node020,node026,node021,node028,node094,node093,node098,node094,node023,node028,node097
#SBATCH -t 30:00:00
#SBATCH --partition=use-everything
#SBATCH -D /om2/user/vanessad/understanding_reasoning/experiment_1/output_slurm_neurips

module add openmind/singularity/3.4.1
hostname


echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3  \
 /om2/user/vanessad/understanding_reasoning/experiment_1/main.py \
--host_filesystem om2 \
--offset_index 0 \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/results_NeurIPS_revision \
--run train \
--experiment_index ${SLURM_ARRAY_TASK_ID}

# --load_model True \
# --module_per_subtask True \
# results_NeurIPS_revision_trial2 wo module_per_subtask