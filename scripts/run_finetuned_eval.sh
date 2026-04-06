#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="data/benchmark/finetuned_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date +%H:%M:%S)] Fine-tuned model evaluation ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --skip-baselines \
    --n-attempts 3

echo "=== [$(date +%H:%M:%S)] Done ==="
echo ""
echo "Results:"
cat data/benchmark/results/finetuned*.json 2>/dev/null | python3 -m json.tool
