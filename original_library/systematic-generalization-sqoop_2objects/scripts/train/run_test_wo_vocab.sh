#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=test_eval_model
#SBATCH --mem=60GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add clustername/singularity/3.4.1

array_string=("17334841.pt.best" "17334848.pt.best" "17334849.pt.best" "17334842.pt.best" "17334847.pt.best")

for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv path_to_singularity-tensorflow-latest-tqm.simg \
    python /path_to_folder/path_to_folder/compositionality/scripts/run_model.py \
    --program_generator $value \
    --execution_engine $value \
    --data_dir /path_to_folder/path_to_folder/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

