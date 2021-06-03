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

declare -a StringArray=("and_mat_spa_val" "compare_mat_spa_val" "compare_mat_val" "embed_mat_spa_val" "embed_spa_mat_val" "or_mat_spa_val" "or_mat_val")

for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_0  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/closure_vector_sep_stem/vector_sep_stem_0_output_${val}.h5 \
  --test_dataset CLOSURE
done


for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_1  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/closure_vector_sep_stem/vector_sep_stem_1_output_${val}.h5 \
  --test_dataset CLOSURE
done



for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_2  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/closure_vector_sep_stem/vector_sep_stem_2_output_${val}.h5 \
  --test_dataset CLOSURE
done


for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_3  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/closure_vector_sep_stem/vector_sep_stem_3_output_${val}.h5 \
  --test_dataset CLOSURE
done


for val in "${StringArray[@]}"
do
  singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
  -m scripts.run_model \
  --execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/vector_sep_stem_4  \
  --data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset/CLEVR_v1.0 \
  --part $val \
  --output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CLEVR/closure_vector_sep_stem/vector_sep_stem_4_output_${val}.h5 \
  --test_dataset CLOSURE
done
