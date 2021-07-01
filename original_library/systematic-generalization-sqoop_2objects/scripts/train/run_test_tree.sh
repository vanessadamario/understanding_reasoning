#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=asym
#SBATCH --mem=30GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=normal

module add clustername/singularity/3.4.1
# h5py version <= 2.10.0 required for these experiments

array_string=('18160381_1_.pt.best' '18170768_3_.pt.best' '18170765_1_.pt.best' '18170766_2_.pt.best' '18160380_0_.pt.best')

for value in ${array_string[*]}
do
  singularity exec -B /om:/om --nv path_to_singularity-tensorflow-latest-tqm.simg \
    python /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/scripts/run_model.py \
    --execution_engine /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/tree_mixed/$value \
    --program_generator /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/tree_mixed/$value \
    --output_h5 /path_to_folder/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/tree_mixed/output_$value.h5 \
    --data_dir /path_to_folder/path_to_folder/compositionality/sqoop-no_crowding-variety_1-repeats_30000  \
    --part test
done

