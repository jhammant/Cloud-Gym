#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="data/benchmark/remaining_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  Remaining evaluations"
echo "  Started: $(date)"
echo "============================================"

# Fine-tuned model first
echo ""
echo "=== [$(date +%H:%M:%S)] Fine-tuned model evaluation ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --skip-baselines \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] Fine-tuned eval done ==="

# Then 32b baseline
echo ""
echo "=== [$(date +%H:%M:%S)] Baseline: qwen2.5-coder:32b ==="
python scripts/run_baselines.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --models "qwen2.5-coder:32b" \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] qwen2.5-coder:32b done ==="

echo ""
echo "============================================"
echo "  All done! $(date)"
echo "============================================"
ls -la data/benchmark/results/*.json
