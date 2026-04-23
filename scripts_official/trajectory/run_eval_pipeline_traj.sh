#!/bin/bash
# Trajectory eval pipeline (paper-release).
#
# Runs experiments.eval to generate samples from a trained trajectory model.
# Plain bash runner; add scheduler directives (e.g. #SBATCH) or wrap in your
# own submission template as needed.
#
# Usage:
#   bash scripts_official/trajectory/run_eval_pipeline_traj.sh \
#       <CONFIG_PATH> <CKPT_PATH> <GEN_PATH>

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <CONFIG_PATH> <CKPT_PATH> <GEN_PATH>" >&2
    exit 1
fi

CONFIG_PATH=$1   # config used to train the model
CKPT_PATH=$2     # ckpt to evaluate
GEN_PATH=$3      # path where generations will be written

# Multi-GPU NCCL timeouts for long runs:
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

echo "Running on node: $(hostname)"

echo "Starting inference..."
python -m experiments.eval \
  --config "$CONFIG_PATH" \
  --checkpoint "$CKPT_PATH" \
  --gen_path "$GEN_PATH"

echo "Done"
