#!/usr/bin/env bash
# Usage: scripts/run_dgx.sh [GPU_INDEX]   (defaults to GPU 0)
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m mltok sysinfo
python -m mltok selftest --single
python -m mltok ingest   -c configs/dgx_full.yaml
python -m mltok clean    -c configs/dgx_full.yaml
# the run waits for >=60 GB free on the GPU and caps itself at 60 GB (see configs/dgx_full.yaml -> gpu)
python -m mltok run      -c configs/dgx_full.yaml --set gpu.index=$GPU
python -m mltok verify   --tokenizer "$(python - <<'PY'
import yaml;c=yaml.safe_load(open('configs/dgx_full.yaml'));print(c['paths']['runs']+'/'+c['name']+'/best')
PY
)"
