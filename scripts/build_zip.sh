#!/usr/bin/env bash
# Build the zip you upload as a Kaggle Dataset (or just `git push` the repo and clone it in the notebook).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-../mltok-pipeline.zip}"
rm -f "$OUT"
zip -r "$OUT" . -x '*/__pycache__/*' '*.pyc' '.pytest_cache/*' 'runs/*' '*.egg-info/*' > /dev/null
echo "wrote $OUT"
