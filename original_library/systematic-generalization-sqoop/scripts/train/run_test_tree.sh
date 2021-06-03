#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=test_eval_model
#SBATCH --mem=10GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

array_string=("17334841.pt.best" "17334848.pt.best" "17334849.pt.best" "17334842.pt.best" "17334847.pt.best")
for value in ${array_string[*]}
do
  singularity exec -B /om:/om --nv path_to_singularity_containers \
    python path_to_folder_repeated/original_library/systematic-generalization-sqoop/scripts/run_model.py \
    --execution_engine $value \
    --program_generator $value \ 
    --vocab_json path_to_folder_repeated/compositionality/sqoop-variety_1-repeats_30000/vocab.json \
    --data_dir path_to_folder_repeated/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

# tensorflow-latest-tqm.simg

