#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=eval_test1
#SBATCH --array=0
#SBATCH --mem=12GB
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:titan-x:1
#SBATCH --partition=normal

module add cluster/singularity/3.4.1

for i in {0..279}
do
  singularity exec -B /om2:/om2 --nv path_singularity-latest-tqm.simg python3 main.py \
  --host_filesystem om2 \
  --experiment_index $i \
  --output_path path_folder/understanding_reasoning/experiment_1/comparison_early_stopping/ \
  --run test
done

# --test_oos 1 \
# --on_validation True \