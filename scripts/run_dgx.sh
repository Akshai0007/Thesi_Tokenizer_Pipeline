#!/usr/bin/env bash
# Usage: scripts/run_dgx.sh [GPU_INDEX] [CONFIG]      defaults: GPU 0, configs/dgx_confirm.yaml
#   step 1: scripts/run_dgx.sh 0 configs/dgx_confirm.yaml    (confirm 48K vs 64K and Unigram vs best BPE at scale)
#   step 2: scripts/run_dgx.sh 0 configs/dgx_final.yaml      (train the final tokenizer)
# Both steps can be started at the same time (e.g. two tmux windows): the download is serialized with a lock and done once,
# cleaned data and run folders are separate. Re-running is cheap: finished downloads/cleaning/results are reused (FORCE_CLEAN=1 re-cleans).
# Optional checkpoint store (pushes every finished unit to a NEW private Hugging Face repo, restores on restart):
#   export HF_TOKEN=...            # write access (also needed for dataset downloads)
#   export HF_REPO=Akzhai/mltok-malayalam-runs
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
CFG="${2:-configs/dgx_confirm.yaml}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cfgval() { python -c "import yaml,sys;c=yaml.safe_load(open('$CFG'));print(eval(sys.argv[1]))" "$1"; }
RAW="$(cfgval "c['paths']['raw']")"
CLEAN="$(cfgval "c['paths']['clean']")"
RUNS="$(cfgval "c['paths']['runs']+'/'+c['name']")"
USES_GPU="$(cfgval "bool(c.get('proxy',{}).get('enabled', False))")"
SYNC_DATA="$(cfgval "bool(c.get('sync',{}).get('include_data', False))")"

STORE_SET=""
if [ -n "${HF_REPO:-}" ]; then
  STORE_SET="store={\"type\":\"hf_hub\",\"repo\":\"${HF_REPO}\",\"prefix\":\"dgx\"}"
fi

python -m mltok sysinfo
if [ "$USES_GPU" = "True" ]; then python -m mltok selftest --single; fi
if [ -n "$STORE_SET" ]; then
  python -m mltok store-check -c "$CFG" --set "$STORE_SET"      # fail early if the token/repo does not work
  RESTORE_SCOPE=run
  if [ "$SYNC_DATA" = "True" ]; then RESTORE_SCOPE=all; fi
  if [ ! -d "$RUNS" ] || [ -z "$(ls -A "$RUNS")" ]; then
    python -m mltok restore -c "$CFG" --set "$STORE_SET" --what "$RESTORE_SCOPE"
  fi
fi

mkdir -p "$RAW"
flock "$RAW/.ingest.lock" python -m mltok ingest -c "$CFG"       # serialized: a second script waits, then finds the files and skips
if [ -f "$CLEAN/clean_stats.json" ] && [ ! -e "$CLEAN/.mltok-cleaning" ] && [ "${FORCE_CLEAN:-0}" != "1" ]; then
  echo "[run_dgx] cleaned data already present in $CLEAN - skipping clean (FORCE_CLEAN=1 to redo)"
else
  if [ -n "$STORE_SET" ]; then
    python -m mltok clean -c "$CFG" --set "$STORE_SET"
  else
    python -m mltok clean -c "$CFG"
  fi
fi

# the run waits for >= 60 GB free on the GPU and caps itself at 60 GB (see `gpu:` in the config)
if [ -n "$STORE_SET" ]; then
  python -m mltok run -c "$CFG" --set "gpu.index=$GPU" "$STORE_SET"
else
  python -m mltok run -c "$CFG" --set "gpu.index=$GPU"
fi
python -m mltok verify --tokenizer "$RUNS/best"
