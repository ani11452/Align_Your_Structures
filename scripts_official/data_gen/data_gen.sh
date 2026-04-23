#!/bin/bash
# MD trajectory generation (paper-release).
#
# Generates train / val / test MD trajectories for a given dataset on a
# single GPU. Launches `generate_data.py` in parallel waves of up to 3
# python processes (each spawning 3 openmm workers) to stay within a
# typical 11-CPU / 1-GPU allocation.
#
# Usage:
#   bash scripts_official/data_gen/data_gen.sh <qm9|drugs> <gpu_id>
#
# Parallel fanout across N GPUs (same host):
#   for id in {0..9}; do bash scripts_official/data_gen/data_gen.sh qm9 $id & done
#   wait
#
# Env:
#   MD_DATA_ROOT   path prefix that contains processed_input_data/ and
#                  output_trajectories/

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <qm9|drugs> <gpu_id>" >&2
    exit 1
fi

DATASET=$1
GPU_ID=$2

case "${DATASET}" in
    qm9)   GEOM=GEOM-QM9   ;;
    drugs) GEOM=GEOM-DRUGS ;;
    *) echo "Unknown dataset: ${DATASET} (expected qm9 or drugs)" >&2; exit 1 ;;
esac

if [ -z "${MD_DATA_ROOT:-}" ]; then
    echo "MD_DATA_ROOT is not set" >&2
    exit 1
fi

DATA_PKL_DIR="${MD_DATA_ROOT}/processed_input_data/${GEOM}"
OUTDIR_BASE="${MD_DATA_ROOT}/output_trajectories/${GEOM}/4fs_HMR15_5ns_actual"

run_split() {
    local split=$1            # train | val | test
    local percentage_flag=$2  # e.g. "--percentage 0.12"  (empty for test)
    local data_pkl="${DATA_PKL_DIR}/${GEOM}_$(tr '[:lower:]' '[:upper:]' <<< ${split:0:1})${split:1}.pkl"
    local outdir="${OUTDIR_BASE}/${split}"

    # Wave 1: ids 0..2  (3 processes -> 9 workers)
    for id in {0..2}; do
        process_id=$(( GPU_ID * 5 + id ))
        echo "Launching ${split^} Generative Process: $process_id"
        python data_gen/simulate/generate_data.py \
            --data_pkl "${data_pkl}" \
            --outdir "${outdir}" \
            --dataset "${DATASET}" --split "${split}" \
            ${percentage_flag} \
            --inference_id "$process_id" --num_inferences 160 --num_workers 3 \
            --hydrogenMass 1.5 \
            --sim_ns 5 --dt 4 --equilibration_steps 5000 --frame_interval 100 &
    done
    wait

    # Wave 2: ids 3..4  (2 processes -> 6 workers)
    for id in {3..4}; do
        process_id=$(( GPU_ID * 5 + id ))
        echo "Launching ${split^} Generative Process: $process_id"
        python data_gen/simulate/generate_data.py \
            --data_pkl "${data_pkl}" \
            --outdir "${outdir}" \
            --dataset "${DATASET}" --split "${split}" \
            ${percentage_flag} \
            --inference_id "$process_id" --num_inferences 160 --num_workers 3 \
            --hydrogenMass 1.5 \
            --sim_ns 5 --dt 4 --equilibration_steps 5000 --frame_interval 100 &
    done
    wait
}

# 20K train trajectories from 4K unique molecules (~12%)
run_split train "--percentage 0.12"

# 5K val trajectories from 1K unique molecules (~22%)
run_split val "--percentage 0.22"

# 5K test trajectories from 1K unique molecules
# (all molecules represented, ~7% of conformers)
run_split test ""
