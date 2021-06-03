#!/bin/bash
#SBATCH -N 1
#SBATCH --array=1-4
#SBATCH -c 1
#SBATCH --job-name=test_CLEVR
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 02:00:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/CLOSURE-master/output_slurm

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd /om2/user/vanessad/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_${SLURM_ARRAY_TASK_ID} \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_${SLURM_ARRAY_TASK_ID}_output_val.h5 \
--test_dataset CLEVR