#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$project_root/data"

usage() {
  cat <<'EOF'
Usage: scripts/dataset_download.sh HF_REPO_ID [DATASET_NAME] [REVISION]

Download a Hugging Face dataset repository into data/, preserving repository
paths and filenames. By default the whole repository is downloaded. Pass a
single subdirectory name to download only that dataset into data/NAME.
REVISION is optional and defaults to main. Pass an empty DATASET_NAME to set
REVISION without selecting a subdirectory.
Downloaded files can update matching paths already under data/. Hugging Face
also stores download metadata in data/.cache/huggingface/.

Examples:
  scripts/dataset_download.sh username/robot-datasets
  scripts/dataset_download.sh username/robot-datasets forte_heuristic_final_100hz
  scripts/dataset_download.sh username/robot-datasets '' v2

Requires the Hugging Face CLI (`hf`; install with `pip install -U
huggingface_hub`). Access to private repositories requires an authenticated
account with read access (`hf auth login`). This script never logs in.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

repo_id="$1"
dataset_name="${2:-}"
revision="${3:-main}"
if [[ -n "$dataset_name" && ( "$dataset_name" == */* || "$dataset_name" == "." || "$dataset_name" == ".." ) ]]; then
  printf 'DATASET_NAME must be a single directory name under data/.\n' >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  printf 'Hugging Face CLI "hf" was not found on PATH.\n' >&2
  exit 1
fi

args=(download "$repo_id" --repo-type dataset --local-dir "$data_dir" --revision "$revision")
if [[ -n "$dataset_name" ]]; then
  args+=(--include "$dataset_name/**")
fi
hf "${args[@]}"
