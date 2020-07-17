#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=MAC_8lhs
#SBATCH --mem=12GB
#SBATCH -t 20:30:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/om/user/vanessad/original_library/systematic-generalization-sqoop/scripts/train_model.py \
  --feature_dim=3,64,64 \
  --model_type MAC \
  --data_dir /om/user/vanessad/om/user/vanessad/compositionality/sqoop-variety_8-repeats_3750 \
  --num_iterations 100000 \
  --checkpoint_every 1000 \
  --record_loss_every 10 \
  --num_val_samples 1000 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --batch_size 128 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 6 \
  --module_stem_subsample_layers 1,3 \
  --module_stem_kernel_size 3 \
  --mac_question_embedding_dropout 0. \
  --mac_stem_dropout 0. \
  --mac_memory_dropout 0. \
  --mac_read_dropout 0. \
  --mac_use_prior_control_in_control_unit 0 \
  --mac_embedding_uniform_boundary 1.0 \
  --mac_nonlinearity ReLU \
  --variational_embedding_dropout 0. \
  --module_dim 128 \
  --num_modules 12 \
  --mac_use_self_attention 0 \
  --mac_use_memory_gate 0 \
  --bidirectional 1 \
  --encoder_type lstm \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 300 \
  --rnn_hidden_dim 128 \
  --rnn_dropout 0 \
  --rnn_output_batchnorm 0 \
  --classifier_fc_dims 1024 \
  --classifier_batchnorm 0 \
  --classifier_dropout 0. \
  --use_local_copies 0 \
  --grad_clip 8. \
  --program_generator_parameter_efficient 1 $@
