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
Remote repository access is checked before downloading so authentication or
revision errors cannot be mistaken for a successful download of local files.

Examples:
  scripts/dataset_download.sh username/robot-datasets
  scripts/dataset_download.sh username/robot-datasets forte_heuristic_final_100hz
  scripts/dataset_download.sh username/robot-datasets '' v2

Requires the Hugging Face CLI (`hf`; install with `pip install -U
huggingface_hub`) and HUGGINGFACE_API_KEY in the project-root .env file,
using shell-compatible KEY=value assignments. The token must have read
access to the dataset. It is passed to the CLI through HF_TOKEN; no
`hf auth login` is needed.
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

env_file="$project_root/.env"
if [[ ! -f "$env_file" ]]; then
  printf 'Hugging Face authentication requires %s.\n' "$env_file" >&2
  exit 1
fi
HUGGINGFACE_API_KEY=""
source "$env_file"
if [[ -z "${HUGGINGFACE_API_KEY:-}" ]]; then
  printf 'Set HUGGINGFACE_API_KEY in %s.\n' "$env_file" >&2
  exit 1
fi
export HF_TOKEN="$HUGGINGFACE_API_KEY"

# snapshot_download can return an existing local directory after a remote error.
if ! hf datasets info "$repo_id" --revision "$revision" >/dev/null; then
  printf 'Cannot access dataset %s at revision %s. Check HUGGINGFACE_API_KEY in %s and repository access.\n' \
    "$repo_id" "$revision" "$env_file" >&2
  exit 1
fi

args=(download "$repo_id" --repo-type dataset --local-dir "$data_dir" --revision "$revision")
if [[ -n "$dataset_name" ]]; then
  args+=(--include "$dataset_name/**")
fi
hf "${args[@]}"
