#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -c 1
#SBATCH --job-name=test_sep
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH -t 01:00:00
#SBATCH -x node023,node026,node035
#SBATCH --partition=normal
#SBATCH -D path_to_folder/understanding_reasoning/CLOSURE-master/output_slurm

module add clustername/singularity/3.4.1
hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

cd path_to_folder/understanding_reasoning/CLOSURE-master

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_0  \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_0_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_0 \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_0_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_1  \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_1_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_1 \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_1_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_2  \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_2_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_2 \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_2_output_valB.h5 \
--test_dataset CoGenT


singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_3  \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'val' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_3_output_valA.h5 \
--test_dataset CoGenT

singularity exec -B /om2:/om2 --nv path_to_singularity python3 \
-m scripts.run_model \
--execution_engine path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_3 \
--data_dir path_to_folder/understanding_reasoning/CLOSURE-master/dataset_visual_bias \
--part 'valB' \
--output_h5 path_to_folder/understanding_reasoning/CLOSURE-master/results/CoGenT/simpleModule_noFiLM_3_output_valB.h5 \
--test_dataset CoGenT

# img tensorflow2.5.0