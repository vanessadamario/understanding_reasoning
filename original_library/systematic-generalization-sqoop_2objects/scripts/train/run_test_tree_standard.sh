#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=testSQOOP
#SBATCH --mem=30GB
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
# h5py version <= 2.10.0 required for these experiments

array_string=('17475752_4_.pt' '17475753_0_.pt' '17475754_1_.pt' '17475755_2_.pt' '17475756_3_.pt')

for value in ${array_string[*]}
do
  singularity exec -B /om:/om --nv path_to_singularity-tensorflow2.5.0.simg \
    python3 /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/scripts/run_model.py \
    --execution_engine /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/_tree_models_2objs/find/lhs1/$value \
    --program_generator /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/_tree_models_2objs/find/lhs1/$value \
    --output_h5 /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/_tree_models_2objs/find/lhs1/output_$value.h5 \
    --data_dir /path_to_folder/path_to_folder/compositionality/sqoop-no_crowding-variety_1-repeats_30000  \
    --part test
done

