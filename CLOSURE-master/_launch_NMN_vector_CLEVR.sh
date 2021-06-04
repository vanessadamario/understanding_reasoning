#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-1,3-4
#SBATCH -c 1
#SBATCH --job-name=vecBNClevr
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 150:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
hostname

echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd path_to_folder/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.train_model \
--model_type EE \
--num_iterations 500000 \
--num_val_samples 100000 \
--load_features 0 \
--loader_num_workers 1 \
--record_loss_every 100 \
--learning_rate 1e-4 \
--module_stem_batchnorm 1 \
--classifier_batchnorm 1 \
--classifier_downsample=none \
--classifier_fc_dims= \
--classifier_proj_dim=0 \
--discriminator_downsample=none \
--discriminator_fc_dims= \
--discriminator_proj_dim=0 \
--nmn_use_film=1 \
--nmn_module_pool=max \
--module_num_layers=2 \
--nmn_use_gammas=tanh \
--classifier_fc_dims=1024 \
--batch_size 128 \
--allow_resume True \
--checkpoint_path path_to_folder/understanding_reasoning/CLOSURE-master/results/CLEVR/with_bn/vector_${SLURM_ARRAY_TASK_ID} \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
$@

# ${SLURM_ARRAY_TASK_ID}