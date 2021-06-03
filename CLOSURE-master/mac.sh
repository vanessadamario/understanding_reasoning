#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=MACCoGenT
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 150:00:00
#SBATCH --partition=normal
#SBATCH -D path_to_folder/understanding_reasoning/CLOSURE-master/output_slurm



module add clustername/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd path_to_folder/understanding_reasoning/CLOSURE-master


singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.train_model \
  --model_type MAC \
  --num_iterations 1000000 \
  --print_verbose_every 20000000 \
  --checkpoint_every 500 \
  --record_loss_every 100 \
  --num_val_samples 149991 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --batch_size 64 \
  --use_coords 1 \
  --module_stem_batchnorm 0 \
  --module_stem_num_layers 2 \
  --module_stem_kernel_size 3 \
  --mac_question_embedding_dropout 0. \
  --mac_stem_dropout 0. \
  --mac_memory_dropout 0. \
  --mac_read_dropout 0. \
  --mac_use_prior_control_in_control_unit 0 \
  --variational_embedding_dropout 0. \
  --module_dim 512 \
  --num_modules 12 \
  --mac_use_self_attention 0 \
  --mac_use_memory_gate 0 \
  --bidirectional 1 \
  --encoder_type lstm \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 300 \
  --rnn_hidden_dim 512 \
  --rnn_dropout 0 \
  --rnn_output_batchnorm 0 \
  --classifier_fc_dims 512 \
  --classifier_batchnorm 0 \
  --classifier_dropout 0. \
  --grad_clip 8. \
  --allow_resume True \
  --checkpoint_path path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/MAC_${SLURM_ARRAY_TASK_ID} \
  --data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
  --program_generator_parameter_efficient 1 $@

  # tensorflow2.5.0.simg