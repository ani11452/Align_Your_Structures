#!/bin/bash
# Conformer eval pipeline (paper-release).
#
# Runs experiments.eval -> eval.covmat.covmat_local for a given config/ckpt.
# Plain bash runner; add your scheduler directives (e.g. #SBATCH / #PBS) at
# the top of this file if needed, or wrap in your own submission template.
#
# Usage:
#   bash scripts_official/conformer/run_eval_pipeline_conf.sh \
#       <CONFIG_PATH> <CKPT_PATH> <GEN_PATH> <EVAL_OUTPUT>

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <CONFIG_PATH> <CKPT_PATH> <GEN_PATH> <EVAL_OUTPUT>" >&2
    exit 1
fi

CONFIG_PATH=$1   # config used to train the model
CKPT_PATH=$2     # ckpt to evaluate
GEN_PATH=$3      # path where generations will be written
EVAL_OUTPUT=$4   # path where eval results will be written

echo "Running on node: $(hostname)"

# Step 1: Run inference
echo "Starting inference..."
python -m experiments.eval \
  --config "$CONFIG_PATH" \
  --checkpoint "$CKPT_PATH" \
  --gen_path "$GEN_PATH"

# Step 2: Run evaluation
echo "Running evaluation..."
python -m eval.covmat.covmat_local \
  --config "$CONFIG_PATH" \
  --gen_path "$GEN_PATH" \
  --output_file "$EVAL_OUTPUT"

echo "Done"
