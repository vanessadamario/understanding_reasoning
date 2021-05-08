#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=test_vector
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 03:00:00
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understanding_reasoning/CLOSURE-master/output_slurm

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd /om2/user/vanessad/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_output_valA.h5

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_output_valB.h5


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_0  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_0_output_valA.h5

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_0  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_0_output_valB.h5


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_1  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_1_output_valA.h5

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_1  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_1_output_valB.h5


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_2  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_2_output_valA.h5

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_2  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_2_output_valB.h5


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_3  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_3_output_valA.h5

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_3  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/vector_3_output_valB.h5
