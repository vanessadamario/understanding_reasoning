#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=test_eval_model
#SBATCH --mem=30GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
# h5py version <= 2.10.0 required for these experiments

array_string=("17359950_4.pt" "17359951_0.pt" "17359952_1.pt" "17359953_2.pt" "17359954_3.pt")

for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv path_to_singularity_tensorflow-latest-tqm.simg \
    python3 /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/scripts/run_model.py \
    --execution_engine /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/MAC_100kIters/$value \
    --program_generator /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/MAC_100kIters/$value \
    --output_h5 /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop/results/MAC_100kIters/output_$value.h5 \
    --data_dir /path_to_folder/path_to_folder/compositionality/sqoop-no_crowding-variety_1-repeats_30000 \
    --part test
done

