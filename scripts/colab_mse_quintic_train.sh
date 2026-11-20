#!/usr/bin/env bash
# Train MSE on the constrained-quintic PD collection with reference-velocity tracking.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$root/scripts/colab_flow_train.sh" \
  --dataset "$root/data/forte_pd_quintic_hover5mm_mixed_1" \
  --policy mse --match-dataset-controller "$@"
