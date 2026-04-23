#!/bin/bash
# Trajectory training launcher (paper-release).
#
# Runs trajectory training from a YAML config. Plain bash runner; prepend your
# scheduler directives (e.g. #SBATCH / #PBS) or wrap in your own submission
# template as needed.
#
# Usage:
#   bash scripts_official/trajectory/launch_slurm_traj.sh <CONFIG_PATH>
#
# Example:
#   bash scripts_official/trajectory/launch_slurm_traj.sh \
#       configs_official/trajectory/qm9/qm9_noH_1000_kabsch_traj_interpolator_pretrain.yaml

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <CONFIG_PATH>" >&2
    exit 1
fi
CONFIG_PATH=$1

echo "Running on node: $(hostname)"
echo "Config: ${CONFIG_PATH}"

python -m experiments.train --config "${CONFIG_PATH}"
