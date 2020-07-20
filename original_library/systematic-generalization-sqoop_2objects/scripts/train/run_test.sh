#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0
#SBATCH --job-name=test_eval_model
#SBATCH --mem=60GB
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

array_string=("17335027.pt.best" "17335023.pt.best" "17335025.pt.best" "17335026.pt.best" "17335024.pt.best")
for value in ${array_string[*]}
 do
  singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-latest-tqm.simg \
    python /om/user/vanessad/om/user/vanessad/compositionality/scripts/run_model.py \
    --program_generator $value \
    --execution_engine $value \
    --data_dir /om/user/vanessad/om/user/vanessad/compositionality/sqoop-variety_1-repeats_30000 \
    --part test
done

