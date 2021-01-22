#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=asym
#SBATCH --mem=30GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

array_string=('18160381_1_.pt.best' '18170768_3_.pt.best' '18170765_1_.pt.best' '18170766_2_.pt.best' '18160380_0_.pt.best')

for value in ${array_string[*]}
do
  singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg \
    python /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/scripts/run_model.py \
    --execution_engine /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/tree_mixed/$value \
    --program_generator /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/tree_mixed/$value \
    --output_h5 /om/user/vanessad/understanding_reasoning/original_library/systematic-generalization-sqoop_2objects/results/tree_mixed/output_$value.h5 \
    --data_dir /om/user/vanessad/om/user/vanessad/compositionality/sqoop-no_crowding-variety_1-repeats_30000  \
    --part test
done

