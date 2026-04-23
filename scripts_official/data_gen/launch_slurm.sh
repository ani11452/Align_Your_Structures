#!/bin/bash
# Example SLURM submission: one-GPU data-gen job.
# Adapt the cluster-specific directives below for your environment.
#
# #SBATCH --account=<YOUR_ACCOUNT>
# #SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=normal
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Activate your Python env (adapt path / env name):
# source ~/.bashrc
# conda activate <your_env>

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: sbatch $0 <gpu_id> <qm9|drugs>" >&2
    exit 1
fi

GPU_ID=$1
DATASET=$2

bash scripts_official/data_gen/data_gen.sh "${DATASET}" "${GPU_ID}"

echo "Done"
