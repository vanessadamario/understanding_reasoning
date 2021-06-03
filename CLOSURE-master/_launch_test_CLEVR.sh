#!/bin/bash
#SBATCH -N 1
#SBATCH --array=1-4
#SBATCH -c 1
#SBATCH --job-name=test_CLEVR
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 02:00:00
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd path_to_folder/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_${SLURM_ARRAY_TASK_ID} \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
--part 'val' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_${SLURM_ARRAY_TASK_ID}_output_val.h5 \
--test_dataset CLEVR