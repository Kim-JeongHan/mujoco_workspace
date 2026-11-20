#!/usr/bin/env bash
# Execute the uploaded BC experiment inside the Colab runtime.
set +x
set -Eeuo pipefail

[[ ${MLAB_RUN_ID:-} =~ ^(flow|mse)-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$ ]] || exit 2
[[ ${MLAB_CHUNKS:-} =~ ^[0-9]+$ ]] && ((MLAB_CHUNKS > 0)) || exit 2
[[ ${MLAB_SHA256:-} =~ ^[a-f0-9]{64}$ ]] || exit 2
wandb_mode=${MLAB_WANDB_MODE:-online}
case "$wandb_mode" in online|disabled) ;; *) exit 2 ;; esac
policy=${MLAB_POLICY:-flow}
case "$policy" in mse|flow) ;; *) exit 2 ;; esac
dataset=${MLAB_DATASET:-data/forte_heuristic_final_100hz}
[[ $dataset =~ ^data/[a-zA-Z0-9_-]+$ ]] || exit 2
match_controller=${MLAB_MATCH_DATASET_CONTROLLER:-0}
case "$match_controller" in 0|1) ;; *) exit 2 ;; esac
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
base="/content/$MLAB_RUN_ID"
key_file="$base-wandb-key"
workspace="$base"
result="$workspace/result"
mkdir -p -- "$result"
if [[ -f /usr/lib64-nvidia/libEGL_nvidia.so.0 ]]; then
  # Colab may ship the NVIDIA libraries without a GLVND vendor registration.
  # Select that driver explicitly instead of Mesa's CPU renderer.
  printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0"}}' \
    > "$workspace/nvidia-egl.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$workspace/nvidia-egl.json"
fi
log="$result/console.log"
: > "$log"
finish() {
  local status=$?
  trap - EXIT INT TERM
  rm -f -- "$key_file"
  if ((status != 0)); then
    printf 'Worker failed with exit code %d\n' "$status" >> "$log"
  fi
  printf '{"run_id":"%s","returncode":%d}\n' "$MLAB_RUN_ID" "$status" > "$result/status.json"
  tar -C "$workspace" -czf "$base-result.tar.gz.tmp" result
  mv -- "$base-result.tar.gz.tmp" "$base-result.tar.gz"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ $wandb_mode == online ]]; then
  [[ -f $key_file && ! -L $key_file && -s $key_file ]] || { echo "W&B key file is missing" >&2; exit 2; }
  chmod 600 -- "$key_file"
fi

bundle="$base-input.tar.gz"
if [[ ! -f $bundle ]]; then
  : > "$bundle"
  for ((index=0; index<MLAB_CHUNKS; index++)); do
    printf -v suffix '%04d' "$index"
    part="$bundle.part-$suffix"
    cat -- "$part" >> "$bundle"
    rm -- "$part"
  done
fi
printf '%s  %s\n' "$MLAB_SHA256" "$bundle" | sha256sum -c - 2>&1 | tee -a "$log"
project="$workspace/project"
mkdir -p -- "$project"
"${MLAB_PYTHON:-python3}" - "$bundle" "$project" <<'PY' 2>&1 | tee -a "$log"
import sys
import tarfile
from pathlib import PurePosixPath

archive_path, destination = sys.argv[1:]
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    if any(member.name.startswith("/") or ".." in PurePosixPath(member.name).parts or
           not (member.isfile() or member.isdir()) for member in members):
        raise ValueError("Upload archive contains an unsafe entry")
    archive.extractall(destination, filter="data")
PY
"${MLAB_PYTHON:-python3}" -m pip install uv 2>&1 | tee -a "$log"
uv python install 3.12 2>&1 | tee -a "$log"
cd -- "$project"
uv sync --locked --no-dev --extra learning --python 3.12 2>&1 | tee -a "$log"
if [[ $policy == mse ]]; then
  uv run --no-sync python - <<'PY' 2>&1 | tee -a "$log"
import mujoco
import torch
from OpenGL import GL

if not torch.cuda.is_available():
    raise RuntimeError("The Colab MSE experiment requires its allocated GPU")
model = mujoco.MjModel.from_xml_string("<mujoco/>")
with mujoco.Renderer(model, height=64, width=64):
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    print(f"Evaluation renderer: {vendor} / {renderer}")
    if "NVIDIA" not in vendor:
        raise RuntimeError("Evaluation requires NVIDIA EGL instead of CPU rendering")
PY
fi
if ((match_controller)); then
  uv run --no-sync python - "$dataset" "$result" <<'PY' 2>&1 | tee -a "$log"
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from mujoco_lab.assets import ROBOT_PATH
from mujoco_lab.assets.robot.robot import RobotConfig

dataset, output = map(Path, sys.argv[1:])
manifest = json.loads((dataset / "collection_manifest.json").read_text())
if manifest["controller"]["name"] != "pd":
    raise ValueError("Matched collection control currently supports PD datasets")
if manifest["reference"]["name"] not in ("position_hold", "min_jerk"):
    raise ValueError("Unsupported collection reference")
if manifest["desired_velocity"] not in ("zero", "reference_velocity"):
    raise ValueError("Unsupported collection desired_velocity")
# Change only this uploaded runtime's robot settings, leaving the local workspace intact.
path = ROBOT_PATH / manifest["common"]["robot"] / "robot.yaml"
settings = yaml.safe_load(path.read_text())
settings["controller"] = {
    **manifest["controller"], "desired_velocity": manifest["desired_velocity"]
}
settings["reference"] = {
    key: value for key, value in manifest["reference"].items() if key != "interpolation"
}
path.write_text(yaml.safe_dump(settings, sort_keys=False))
robot = RobotConfig.load(path)
shutil.copy2(path, output / "robot.yaml")
shutil.copy2(dataset / "collection_manifest.json", output / "collection_manifest.json")
shutil.copy2(dataset / "splits.json", output / "requested_splits.json")
print(json.dumps({"controller": asdict(robot.controller), "reference": asdict(robot.reference),
                  "desired_velocity": robot.controller.desired_velocity}))
PY
fi
(
  if [[ $wandb_mode == online ]]; then
    WANDB_API_KEY=$(<"$key_file")
    rm -- "$key_file"
    export WANDB_API_KEY
  else
    unset WANDB_API_KEY
  fi
  train_args=(--policy-type "$policy" --data-dir "$dataset" --output-dir "$result/log")
  if [[ $policy == flow ]]; then
    train_args+=(--eval-interval-steps 0)
  else
    train_args+=(--eval-interval-steps 30000 --evaluation.num-episodes 3 --evaluation.video-episodes 3)
  fi
  if ((match_controller)); then
    train_args+=(--split-file "$dataset/splits.json")
  fi
  if [[ -n ${WANDB_NAME:-} ]]; then
    train_args+=(--run-name "$WANDB_NAME")
  fi
  WANDB_MODE="$wandb_mode" uv run --no-sync python -m mujoco_lab.learning.train \
    "${train_args[@]}"
) 2>&1 | tee -a "$log"
