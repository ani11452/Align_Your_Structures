#!/bin/bash
# Conformer training launcher (paper-release).
#
# Runs conformer training from a YAML config. The script is a plain bash
# runner; if you use SLURM / another scheduler, prepend the appropriate
# directives (e.g. #SBATCH / #PBS) to suit your cluster, or wrap this
# script in your own submission template.
#
# Usage:
#   bash scripts_official/conformer/launch_slurm_conf.sh <CONFIG_PATH>
#
# Example:
#   bash scripts_official/conformer/launch_slurm_conf.sh \
#       configs_official/conformer/qm9/qm9_noH_1000_kabsch_conf_basic_es_order_3.yaml

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <CONFIG_PATH>" >&2
    exit 1
fi
CONFIG_PATH=$1

echo "Running on node: $(hostname)"
echo "Config: ${CONFIG_PATH}"

python -m experiments.train --config "${CONFIG_PATH}"
