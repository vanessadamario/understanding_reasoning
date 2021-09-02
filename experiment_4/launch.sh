#!/bin/bash
#SBATCH -N 1
#SBATCH --array=11-12
#SBATCH --job-name=sqoop_4
#SBATCH --mem=3GB
#SBATCH --constraint=8GB
#SBATCH -x node022,node023,node026,node021,node028,node094,node093
#SBATCH --gres=gpu:1
#SBATCH -t 80:00:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/experiment_4/slurm_output

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3 \
/om2/user/vanessad/understanding_reasoning/experiment_4/main.py \
--host_filesystem om2_exp4 \
--offset_index 0 \
--load_model True \
--output_folder sqoop_exps_4 \
--sqoop_dataset True \
--run train \
--experiment_index ${SLURM_ARRAY_TASK_ID}


# ${SLURM_ARRAY_TASK_ID}

# spatial_only
# 181-200,201,202,203,204,205,206,207,208,209 without load
# 5-8,10,11,14,36,40,41,44,66-69,71,73,96-100,102-104,125-134,156-164,185-194,210-246,248,249
