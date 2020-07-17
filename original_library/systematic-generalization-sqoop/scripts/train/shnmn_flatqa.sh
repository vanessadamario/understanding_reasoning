#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=chain_lhs1_5obj
#SBATCH --mem=20GB
#SBATCH -t 15:30:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
python /om/user/vanessad/om/user/vanessad/original_library/systematic-generalization-sqoop/scripts/train_model.py \
  --model_type SHNMN \
  --data_dir /om/user/vanessad/om/user/vanessad/compositionality/sqoop-variety_1-repeats_30000 \
  --hard_code_tau --tau_init chain --hard_code_alpha --alpha_init correct\
  --feature_dim=3,64,64 \
  --num_iterations=50000 \
  --checkpoint_every 1000 \
  --record_loss_every 10 \
  --num_val_samples 1000 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 6 \
  --module_stem_subsample_layers 1,3 \
  --module_intermediate_batchnorm 0 \
  --module_batchnorm 0 \
  --module_dim 64 \
  --classifier_batchnorm 1 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --program_generator_parameter_efficient 1 $@
