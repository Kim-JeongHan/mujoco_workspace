#!/usr/bin/env bash
# Run a local BC experiment on a Colab GPU session.
set +x
set -euo pipefail
umask 077

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dataset="$root/data/forte_heuristic_final_100hz"
gpu=T4
policy=flow
match_controller=0
existing_session=
timeout=86400
wandb_mode=${WANDB_MODE:-online}
usage() {
  echo "Usage: $0 [--dataset PATH] [--policy mse|flow] [--match-dataset-controller] [--gpu T4|L4|G4|A100|H100] [--session NAME] [--timeout SECONDS]"
  echo "--session reuses an existing session and stops it after verified result retrieval."
}
while (($#)); do
  case "$1" in
    --dataset|--gpu|--timeout|--policy|--session)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in
        --dataset) dataset=$2 ;;
        --gpu) gpu=$2 ;;
        --timeout) timeout=$2 ;;
        --policy) policy=$2 ;;
        --session) existing_session=$2 ;;
      esac
      shift 2 ;;
    --match-dataset-controller) match_controller=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$gpu" in T4|L4|G4|A100|H100) ;; *) echo "Unsupported GPU: $gpu" >&2; exit 2 ;; esac
case "$policy" in mse|flow) ;; *) echo "Unsupported policy: $policy" >&2; exit 2 ;; esac
case "$wandb_mode" in online|disabled) ;; *) echo "Unsupported W&B mode: $wandb_mode" >&2; exit 2 ;; esac
if [[ $wandb_mode == online && -z ${WANDB_API_KEY:-} ]]; then
  echo "W&B online mode requires WANDB_API_KEY. Set it in your shell or use WANDB_MODE=disabled." >&2
  exit 2
fi
[[ $timeout =~ ^[0-9]+$ ]] && ((timeout >= 60)) || { echo "Invalid timeout" >&2; exit 2; }
[[ ! -L $dataset && -d $dataset ]] || { echo "Dataset must be a regular directory" >&2; exit 2; }
dataset=$(cd -- "$dataset" && pwd)
dataset_name=${dataset##*/}
[[ $dataset_name =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid dataset directory name" >&2; exit 2; }
if ((match_controller)); then
  [[ -f $dataset/collection_manifest.json && -f $dataset/splits.json ]] || {
    echo "Controller matching requires collection_manifest.json and splits.json" >&2; exit 2;
  }
fi
shopt -s nullglob
episodes=("$dataset"/*.npz)
((${#episodes[@]})) || { echo "Dataset has no .npz episodes: $dataset" >&2; exit 2; }
[[ ! -L $root/src && -d $root/src ]] || { echo "Source directory is missing or a symlink" >&2; exit 2; }
for name in pyproject.toml uv.lock README.md .python-version; do
  [[ -f $root/$name && ! -L $root/$name ]] || { echo "Project file missing or symlink: $name" >&2; exit 2; }
done
if [[ -n $(find "$root/src" "$dataset" -type l -print -quit) ]]; then
  echo "Upload source and dataset cannot contain symlinks" >&2
  exit 2
fi

run_id="$policy-$(date -u +%Y%m%d-%H%M%S)-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
session=${existing_session:-mujoco-$run_id}
remote_base="/content/$run_id"
remote_result="$remote_base-result.tar.gz"
remote_key="$remote_base-wandb-key"
result_dir="$root/log/colab/$run_id"
mkdir -p -- "$root/log/colab"
mkdir -- "$result_dir"
temp_dir=$(mktemp -d -t mujoco-colab-XXXXXXXX)
owned=0
worker_started=0
retrieved=0
key_upload_attempted=0
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  if ((owned)); then
    if ((retrieved || !worker_started)); then
      if ! colab --auth oauth2 stop -s "$session"; then
        echo "Stop failed. Stop manually: colab stop -s $session" >&2
        if ((key_upload_attempted)); then
          colab --auth oauth2 rm -s "$session" "$remote_key" >/dev/null 2>&1 || \
            echo "Could not confirm remote W&B key removal; stop the session promptly." >&2
        fi
        ((exit_status != 0)) || exit_status=1
      fi
    else
      if ((key_upload_attempted)); then
        colab --auth oauth2 rm -s "$session" "$remote_key" >/dev/null 2>&1 || \
          echo "Could not confirm remote W&B key removal; stop the session promptly." >&2
      fi
      echo "Results unverified. Session $session remains active and consumes compute units." >&2
      echo "Recover $remote_result, then run: colab stop -s $session" >&2
    fi
  fi
  rm -rf -- "$temp_dir"
  exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ $wandb_mode == online ]]; then
  printf '%s' "$WANDB_API_KEY" > "$temp_dir/wandb-key"
fi
unset WANDB_API_KEY

# Archive only source files needed by the package and the selected .npz episodes.
file_list="$temp_dir/source-files"
printf 'pyproject.toml\0uv.lock\0README.md\0.python-version\0' > "$file_list"
while IFS= read -r -d '' path; do
  relative=${path#"$root"/}
  case "/$relative/" in */.*/*|*/__pycache__/*) continue ;; esac
  basename=${path##*/}
  case "${basename,,}" in
    *secret*|*credential*|*.pem|*.key|*token*.json|*token*.yaml|*key*.json|*key*.yaml) continue ;;
  esac
  case "$path" in *.py|*.xml|*.stl|*.obj|*.yaml|*.md|*.urdf|*.json|*.txt)
    printf '%s\0' "$relative" >> "$file_list" ;;
  esac
done < <(find "$root/src" -type f -print0)
tar -C "$root" --null -T "$file_list" -cf "$temp_dir/input.tar"
data_list="$temp_dir/data-files"
: > "$data_list"
for path in "${episodes[@]}"; do
  [[ -f $path && ! -L $path ]] || { echo "Episode must be a regular file: $path" >&2; exit 2; }
  printf '%s\0' "${path##*/}" >> "$data_list"
done
for name in collection_manifest.json splits.json; do
  if [[ -f $dataset/$name && ! -L $dataset/$name ]]; then
    printf '%s\0' "$name" >> "$data_list"
  fi
done
tar -C "$dataset" --null -T "$data_list" \
  --transform="s,^,data/$dataset_name/," -rf "$temp_dir/input.tar"
gzip -c "$temp_dir/input.tar" > "$temp_dir/input.tar.gz"
sha256=$(sha256sum < "$temp_dir/input.tar.gz" | awk '{print $1}')
split -b 8m -d -a 4 --additional-suffix=.part \
  "$temp_dir/input.tar.gz" "$temp_dir/chunk-"
parts=("$temp_dir"/chunk-*.part)
chunks=${#parts[@]}
python3 - "$result_dir/session.json" "$session" "$gpu" "$dataset" "$sha256" "$remote_result" "$policy" "$match_controller" <<'PY'
import json
import sys
from pathlib import Path

path, session, gpu, dataset, digest, remote_result, policy, match_controller = sys.argv[1:]
Path(path).write_text(json.dumps({"session": session, "gpu": gpu, "dataset": dataset,
                                  "policy": policy, "match_dataset_controller": bool(int(match_controller)),
                                  "bundle_sha256": digest, "remote_result": remote_result}, indent=2) + "\n")
PY
echo "Prepared $(du -h "$temp_dir/input.tar.gz" | awk '{print $1}') in $chunks upload chunks"
rm -- "$temp_dir/input.tar" "$temp_dir/input.tar.gz"
if [[ -z $existing_session ]]; then
  colab --auth oauth2 new -s "$session" --gpu "$gpu"
fi
owned=1
for index in "${!parts[@]}"; do
  printf -v suffix '%04d' "$index"
  colab --auth oauth2 upload -s "$session" "${parts[index]}" "$remote_base-input.tar.gz.part-$suffix"
  rm -- "${parts[index]}"
done
colab --auth oauth2 upload -s "$session" "$root/scripts/colab_flow_worker.sh" "$remote_base-worker.sh"
if [[ $wandb_mode == online ]]; then
  key_upload_attempted=1
  colab --auth oauth2 upload -s "$session" "$temp_dir/wandb-key" "$remote_key"
  rm -- "$temp_dir/wandb-key"
fi
worker_started=1
env_args=(--env "MLAB_RUN_ID=$run_id" --env "MLAB_CHUNKS=$chunks" \
  --env "MLAB_SHA256=$sha256" --env "MLAB_WANDB_MODE=$wandb_mode" \
  --env "MLAB_POLICY=$policy" --env "MLAB_DATASET=data/$dataset_name" \
  --env "MLAB_MATCH_DATASET_CONTROLLER=$match_controller")
for name in WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP WANDB_NAME WANDB_RUN_ID; do
  if [[ -n ${!name:-} ]]; then
    env_args+=(--env "$name=${!name}")
  fi
done
if ! colab --auth oauth2 exec -s "$session" -f "$root/scripts/colab_flow_worker.py" \
  --timeout "$timeout" "${env_args[@]}"; then
  echo "Colab execution returned an error; trying to retrieve results." >&2
fi
colab --auth oauth2 download -s "$session" "$remote_result" "$temp_dir/result.tar.gz"
status=$(python3 - "$temp_dir/result.tar.gz" "$result_dir" "$run_id" <<'PY'
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive_path, output, run_id = sys.argv[1:]
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    if any(".." in PurePosixPath(m.name).parts or
           not (m.name == "result" or m.name.startswith("result/")) or
           not (m.isfile() or m.isdir()) for m in members):
        raise ValueError("Result archive contains an unsafe entry")
    archive.extractall(output, filter="data")
result = Path(output) / "result"
status = json.loads((result / "status.json").read_text())
if status.get("run_id") != run_id or type(status.get("returncode")) is not int:
    raise ValueError("Result status is missing or does not match this run")
if status["returncode"] == 0 and not list((result / "log").rglob("checkpoint.pt")):
    raise ValueError("Training reported success, but no checkpoint was retrieved")
print(status["returncode"])
PY
)
retrieved=1
if ((status != 0)); then
  echo "Remote training failed; see $result_dir/result/console.log" >&2
  exit 1
fi
echo "Checkpoint downloaded under $result_dir/result/log"
