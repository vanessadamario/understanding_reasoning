#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=extract_feats
#SBATCH --mem=10GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 01:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

# CODE PERFORMS QUESTION GENERATION FOR CLOSURE
# QUESTION GENERATION FOR OTHER CLEVR DATASET (COMMENTED)
# FEATURE EXTRACTION (COMMENTED)
cd path_to_folder/understanding_reasoning/CLOSURE-master

declare -a StringArray=("and_mat_spa_val.json" "compare_mat_spa_val.json" "compare_mat_val.json" "embed_mat_spa_val.json" "embed_spa_mat_val.json" "or_mat_spa_val.json")

for val in "${StringArray[@]}"
do
  echo $val
  singularity exec -B /om2:/om2 --nv singularity_path python3 \
  ./scripts/preprocess_questions.py \
  --input_questions_json path_to_folder/understanding_reasoning/CLOSURE-master/closure_2/${val} # dataset/CLOSURE/$val \
  --output_h5_file path_to_folder/understanding_reasoning/CLOSURE-master/closure_2/${val}_val_questions.h5 # dataset/CLOSURE/${val}_val_questions.h5 \
  --output_vocab_json path_to_folder/understanding_reasoning/CLOSURE-master/closure_2/vocab.json
done
# tensorflow2.5.0.simg
#

# QUESTION GENERATION FOR ORIGINAL CLEVR DATASET
#singularity exec -B /om2:/om2 --nv singularity_path python3 \
#./scripts/preprocess_questions.py \
#--input_questions_json dataset/CLEVR_v1.0/questions/CLEVR_val_questions.json \
#--output_h5_file dataset/val_questions.h5 \
#--output_vocab_json dataset/vocab.json
#tensorflow2
#
#singularity exec -B /om2:/om2 --nv singularity_path python3 \
#./scripts/preprocess_questions.py \
#--input_questions_json dataset/CLEVR_v1.0/questions/CLEVR_test_questions.json \
#--output_h5_file dataset/test_questions.h5 \
#--output_vocab_json dataset/vocab.json
#
# FEATURE EXTRACTION FOR ORIGINAL CLEVR DATASET
## singularity exec -B /om2:/om2 --nv singularity_path python3 \
## ./scripts/extract_features.py \
## --input_image_dir dataset/CLEVR_v1.0/images/train \
## --output_h5_file dataset/CLEVR_v1.0/train_features.h5
#
#
## singularity exec -B /om2:/om2 --nv singularity_path python3 \
## ./scripts/extract_features.py  \
## --input_image_dir dataset/CLEVR_v1.0/images/val \
## --output_h5_file dataset/CLEVR_v1.0/val_features.h5
#
#
## singularity exec -B /om2:/om2 --nv singularity_path python3 \
## ./scripts/extract_features.py  \
## --input_image_dir dataset/CLEVR_v1.0/images/test \
## --output_h5_file dataset/CLEVR_v1.0/test_features.h5
