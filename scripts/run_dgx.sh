#!/usr/bin/env bash
# Usage: scripts/run_dgx.sh [GPU_INDEX]        (defaults to GPU 0)
# Optional checkpoint store (pushes every finished unit to a NEW private Hugging Face repo, restores on restart):
#   export HF_TOKEN=...            # write access
#   export HF_REPO=Akzhai/mltok-malayalam-runs
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
CFG=configs/dgx_full.yaml
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STORE_SET=""
if [ -n "${HF_REPO:-}" ]; then
  STORE_SET="store={\"type\":\"hf_hub\",\"repo\":\"${HF_REPO}\",\"prefix\":\"dgx\"}"
fi

python -m mltok sysinfo
python -m mltok selftest --single
if [ -n "$STORE_SET" ]; then
  python -m mltok store-check -c "$CFG" --set "$STORE_SET"      # fail early if the token/repo does not work
  python -m mltok restore     -c "$CFG" --set "$STORE_SET" --what all
fi
python -m mltok ingest -c "$CFG"
python -m mltok clean  -c "$CFG"
# the run waits for >= 60 GB free on the GPU and caps itself at 60 GB (see `gpu:` in the config)
if [ -n "$STORE_SET" ]; then
  python -m mltok run -c "$CFG" --set "gpu.index=$GPU" "$STORE_SET"
else
  python -m mltok run -c "$CFG" --set "gpu.index=$GPU"
fi
BEST="$(python -c "import yaml;c=yaml.safe_load(open('$CFG'));print(c['paths']['runs']+'/'+c['name']+'/best')")"
python -m mltok verify --tokenizer "$BEST"
