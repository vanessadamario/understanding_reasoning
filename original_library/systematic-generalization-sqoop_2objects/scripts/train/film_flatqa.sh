#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0-4
#SBATCH --job-name=FiLMlhs1
#SBATCH --mem=10GB
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
  python3 /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/scripts/train_model.py \
  --feature_dim=3,64,64 \
  --checkpoint_path /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/FiLM/lhs1/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.pt \
  --data_dir /om/user/vanessad/om/user/vanessad/compositionality/sqoop-no_crowding-variety_1-repeats_30000 \
  --model_type FiLM \
  --num_iterations 200000 \
  --checkpoint_every 1000 \
  --record_loss_every 1 \
  --num_val_samples 1000 \
  --optimizer Adam \
  --learning_rate 3e-4 \
  --batch_size 64 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 6 \
  --module_stem_subsample_layers 1,3\
  --module_batchnorm 1 \
  --classifier_batchnorm 1 \
  --bidirectional 0 \
  --decoder_type linear \
  --encoder_type gru \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 1024 \
  --rnn_output_batchnorm 0 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --classifier_fc_dims 1024 \
  --module_input_proj 1 \
  --module_residual 1 \
  --module_dim 64 \
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
  --condition_method bn-film \
  --program_generator_parameter_efficient 1 $@
