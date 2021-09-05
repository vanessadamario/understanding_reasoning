#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=31
#SBATCH --job-name=NPS_ms2
#SBATCH --mem=40GB
#SBATCH -x node023,node020,node026,node021,node028,node094,node093,node098,node094,node023,node028,node097,dgx001
#SBATCH --gres=gpu:1
#SBATCH -t 35:00:00
#SBATCH --partition=normal
#SBATCH --mail-user=vanessad@mit.edu
#SBATCH --mail-type=ALL

module add openmind/singularity/3.4.1
hostname
singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3 main.py \
--host_filesystem om2 \
--output_path results_NeurIPS_module_per_subtask_trial2 \
--offset_index 0 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--run train

# --load_model True \
# --module_per_subtask True \ this is true for the find per subtask only
# #SBATCH --constraint=any-gpu
# --load_model 1 \