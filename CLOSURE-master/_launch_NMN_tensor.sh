#!/bin/bash
#SBATCH -N 1
#SBATCH --array=1,3,4
#SBATCH -c 1
#SBATCH --job-name=tensorBNCoGenT
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 150:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd path_to_folder/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv path_to_sing python3 \
-m scripts.train_model \
--model_type EE \
--num_iterations 500000 \
--num_val_samples 100000 \
--load_features 0 \
--loader_num_workers 1 \
--record_loss_every 100 \
--module_stem_batchnorm 1 \
--classifier_batchnorm 1 \
--checkpoint_path path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_${SLURM_ARRAY_TASK_ID} \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--learning_rate 1e-4 $@

#tensor_${SLURM_ARRAY_TASK_ID}
# --num_iterations 500000 \
# --num_val_samples 100000 \
