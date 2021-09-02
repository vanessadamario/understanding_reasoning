#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH -c 1
#SBATCH --job-name=vec_sep_NPS
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 120:00:00
#SBATCH --partition=use-everything

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd /om2/user/vanessad/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg python3 \
-m scripts.train_model \
--model_type EE \
--nmn_use_simple_block 0 \
--num_iterations 500000 \
--num_val_samples 100000 \
--load_features 0 \
--loader_num_workers 1 \
--record_loss_every 100 \
--learning_rate 1e-4 \
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
--batch_size 64 \
--allow_resume True \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--separated_stem True \
--separation_per_subtask True \
--checkpoint_path /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT_NeurIPS_revision/vector_sep_stem_${SLURM_ARRAY_TASK_ID} \
$@_

