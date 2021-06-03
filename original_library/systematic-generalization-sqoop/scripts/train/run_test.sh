#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=test_eval_model
#SBATCH --mem=60GB
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

array_string=("17337052.pt" "17337048.pt" "17337050.pt" "17337051.pt" "17337049.pt")
for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv path_to_singularity_containers \
    python /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop/scripts/run_model.py \
    --program_generator path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/$value \
    --execution_engine path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/$value \
    --output_h5 path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/output_$value.h5 \
    --data_dir path_to_folder_repeated/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

array_string=("17329167.pt" "17329163.pt" "17329166.pt" "17329168.pt" "17329164.pt" "17329165.pt")
for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv path_to_singularity_containers \
    python path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/scripts/run_model.py \
    --program_generator path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/$value \
    --execution_engine path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/$value \
    --output_h5 path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/output_$value.h5 \
    --data_dir path_to_folder_repeated/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

array_string=("17334841.pt" "17334848.pt" "17334849.pt" "17334842.pt" "17334847.pt")
for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv path_to_singularity_containers \
    python path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/scripts/run_model.py \
    --program_generator path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/$value \
    --execution_engine path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/$value \
    --output_h5 path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/train/output_$value.h5 \
    --data_dir path_to_folder_repeated/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

array_string=("18162443.pt" "18162444.pt" "18162445.pt" "18162446.pt" "18162447.pt")
for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv path_to_singularity_containers \
    python path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/scripts/run_model.py \
    --program_generator path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/find/$value \
    --execution_engine path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/find/$value \
    --output_h5 path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/find/output_$value.h5 \
    --data_dir path_to_folder/om/user/vanessad/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

