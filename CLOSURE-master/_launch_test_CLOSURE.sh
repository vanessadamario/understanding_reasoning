#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=test_tensor
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

declare -a StringArray=("and_mat_spa_val" "compare_mat_spa_val" "compare_mat_val" "embed_mat_spa_val" "embed_spa_mat_val" "or_mat_spa_val")

for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_0  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_0_output_${val}.h5
done

for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/tensor_0  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/tensor_0_output_${val}.h5
done

for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/FiLM  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0
   \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/FiLM_output_${val}.h5
done