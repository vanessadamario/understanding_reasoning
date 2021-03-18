#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=eval1
#SBATCH --array=256-271,200-207,216-231,80-119,0-7,16-39
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -x node003,node023,node026
#SBATCH -t 01:00:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER
singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3 main.py \
--host_filesystem om2 \
--experiment_index ${SLURM_ARRAY_TASK_ID} \
--output_path /om2/user/vanessad/understanding_reasoning/experiment_1/query_early_stopping/ \
--run test

# ,256-271,200-207,216-231,160-199,80-119,0-7,16-31

# singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 main.py \
# --host_filesystem om2 \
# --experiment_index ${SLURM_ARRAY_TASK_ID} \
# --output_path /om2/user/vanessad/understanding_reasoning/experiment_1/fixstar_results/results/ \
# --new_output_path True \
# --new_data_path True \
# --data_path /om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets \
# --run test

# xboix-tensorflow-latest-tqm.simg
# --test_oos 1 \
# --on_validation True \
# #SBATCH --gres=gpu:1
#SBATCH --constraint=2G
#