#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$project_root/data"

usage() {
  cat <<'EOF'
Usage: scripts/dataset_upload.sh HF_REPO_ID [DATASET_NAME]

Create a private Hugging Face dataset repository if needed, then upload the
contents of data/ to its root, preserving subdirectories. Existing
repositories keep their current visibility.

By default the whole data directory is uploaded. Pass a single subdirectory
name to upload only data/DATASET_NAME, preserving that directory name remotely.
The data/README.md dataset card is uploaded to the repository root in either
mode.
Downloads created with `--local-dir` may add `.cache/huggingface/` metadata;
this script excludes that metadata when uploading.

Examples:
  scripts/dataset_upload.sh username/robot-datasets
  scripts/dataset_upload.sh username/robot-datasets forte_heuristic_final_100hz

Requires the Hugging Face CLI (`hf`; install with `pip install -U
huggingface_hub`) and an authenticated account with write access (`hf auth
login`). The CLI handles authentication; this script never logs in. A newly
created repository is private; an existing repository keeps its visibility.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

repo_id="$1"
dataset_name="${2:-}"
source_dir="$data_dir"
path_in_repo="."
if [[ -n "$dataset_name" ]]; then
  if [[ "$dataset_name" == */* || "$dataset_name" == "." || "$dataset_name" == ".." ]]; then
    printf 'DATASET_NAME must be a single directory name under data/.\n' >&2
    exit 2
  fi
  source_dir="$data_dir/$dataset_name"
  path_in_repo="$dataset_name"
fi

if [[ ! -d "$source_dir" ]]; then
  printf 'Source directory does not exist: %s\n' "$source_dir" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  printf 'Hugging Face CLI "hf" was not found on PATH.\n' >&2
  exit 1
fi

hf repos create "$repo_id" --repo-type dataset --private --exist-ok
hf upload "$repo_id" "$source_dir" "$path_in_repo" --repo-type dataset \
  --exclude '.cache/huggingface/**'

if [[ -n "$dataset_name" ]]; then
  hf upload "$repo_id" "$data_dir/README.md" README.md --repo-type dataset
fi
