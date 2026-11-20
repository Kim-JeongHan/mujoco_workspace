#!/usr/bin/env bash
# Train MSE on the hover-5mm PD minimum-jerk collection with matched rollout control.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$root/scripts/colab_flow_train.sh" \
  --dataset "$root/data/forte_pd_minjerk_hover5mm_mixed_1" \
  --policy mse --match-dataset-controller "$@"
