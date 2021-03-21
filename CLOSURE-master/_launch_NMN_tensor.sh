#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=tensor_cogent
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 60:00:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/CLOSURE-master/output_slurm

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd /om2/user/vanessad/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python3 \
-m scripts.train_model \
--model_type EE \
--num_iterations 500000 \
--num_val_samples 100000 \
--load_features 0 \
--loader_num_workers 1 \
--record_loss_every 100 \
--checkpoint_path /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/tensor \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--learning_rate 1e-4 $@

#tensor_${SLURM_ARRAY_TASK_ID}
# --num_iterations 500000 \
# --num_val_samples 100000 \
