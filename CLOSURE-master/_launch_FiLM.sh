#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0,2
#SBATCH -c 1
#SBATCH --job-name=FiLMCoGenT
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 150:00:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/CLOSURE-master/output_slurm

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd /om2/user/vanessad/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg python3 \
-m scripts.train_model \
--model_type FiLM \
--num_iterations 500000 \
--print_verbose_every 20000000 \
--record_loss_every 100 \
--num_val_samples 149991 \
--optimizer Adam \
--learning_rate 3e-4 \
--batch_size 64 \
--use_coords 1 \
--module_stem_batchnorm 1 \
--module_stem_num_layers 1 \
--module_batchnorm 1 \
--classifier_batchnorm 1 \
--bidirectional 0 \
--decoder_type linear \
--encoder_type gru \
--weight_decay 1e-5 \
--rnn_num_layers 1 \
--rnn_wordvec_dim 200 \
--rnn_hidden_dim 4096 \
--rnn_output_batchnorm 0 \
--classifier_downsample maxpoolfull \
--classifier_proj_dim 512 \
--classifier_fc_dims 1024 \
--module_input_proj 1 \
--module_residual 1 \
--module_dim 128 \
--module_dropout 0e-2 \
--module_stem_kernel_size 3 \
--module_kernel_size 3 \
--module_batchnorm_affine 0 \
--module_num_layers 1 \
--num_modules 4 \
--condition_pattern 1,1,1,1 \
--gamma_option linear \
--gamma_baseline 1 \
--use_gamma 1 \
--use_beta 1 \
--allow_resume True \
--condition_method bn-film \
--checkpoint_path /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/FiLM_${SLURM_ARRAY_TASK_ID} \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--program_generator_parameter_efficient 1 $@
