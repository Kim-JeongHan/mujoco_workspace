#!/usr/bin/env bash
# Train the hover-5mm PD position-hold MSE baseline with matched evaluation control.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$root/scripts/colab_flow_train.sh" \
  --dataset "$root/data/forte_pd_hold_hover5mm_mixed_1" \
  --policy mse --match-dataset-controller "$@"
