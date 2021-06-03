#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=test_tensors
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 05:00:00
#SBATCH -x node023,node026,node035
#SBATCH --partition=cbmm
#SBATCH -D /om2/user/vanessad/understandting_reasoning/CLOSURE-master/output_slurm

module add openmind/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd /om2/user/vanessad/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_0  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_0_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_0 \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_0_value_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_1  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_1_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_1 \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_1_value_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_2  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_2_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_2 \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_2_value_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_3  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_3_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_3 \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_3_value_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_4  \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_4_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg python3 \
-m scripts.run_model \
--execution_engine /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_4 \
--data_dir /om2/user/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 /om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT/with_bn/tensor_4_value_output_valB.h5 \
--test_dataset CoGenT
