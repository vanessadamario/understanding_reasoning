#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0-4
#SBATCH --job-name=mixed_modules
#SBATCH --mem=30GB
#SBATCH -t 40:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal
#SBATCH -D /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/slurm_output
module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv path_to_singularity-tensorflow2.simg \
python /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/scripts/train_model.py \
  --model_type SHNMN \
  --checkpoint_path /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/tree_mixed_find/${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_.pt \
  --data_dir /path_to_folder/path_to_folder/compositionality/sqoop-variety_1-repeats_30000 \
  --hard_code_tau --tau_init tree --hard_code_alpha --alpha_init correct \
  --feature_dim=3,64,64 \
  c500000 \
  --checkpoint_every 500 \
  --use_module=mixed_find \
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
