#!/bin/bash
# Example SLURM fanout: submit N one-GPU data-gen jobs in parallel.
# Adapt to your scheduler as needed.
#
# Usage:
#   bash scripts_official/data_gen/launch_jobs.sh <num_jobs> <qm9|drugs>

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_jobs> <qm9|drugs>" >&2
    exit 1
fi

num_jobs=$1
dataset=$2
ids=$(( num_jobs - 1 ))

case "${dataset}" in
    qm9)   GEOM=GEOM-QM9   ;;
    drugs) GEOM=GEOM-DRUGS ;;
    *) echo "Invalid dataset: ${dataset} (expected qm9 or drugs)" >&2; exit 1 ;;
esac

if [ -z "${MD_DATA_ROOT:-}" ]; then
    echo "MD_DATA_ROOT is not set" >&2
    exit 1
fi

LOGS_DIR="${MD_DATA_ROOT}/output_trajectories/${GEOM}/4fs_HMR15_5ns_actual/job_logs"
mkdir -p "${LOGS_DIR}"

for id in $(seq 0 $ids); do
    echo "Launching SLURM job ${id} of ${ids} (dataset=${dataset})"
    sbatch \
        --job-name="md_${id}" \
        --output="${LOGS_DIR}/md_gpu_${id}.out" \
        scripts_official/data_gen/launch_slurm.sh "${id}" "${dataset}"
    echo "Submitted md_gpu_${id}_of_${ids} for dataset ${dataset}."
done
