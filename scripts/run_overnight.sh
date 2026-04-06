#!/bin/bash
# Cloud-Gym: Overnight evaluation pipeline
# Runs baselines + fine-tuned model eval, logs everything
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate

LOG="data/benchmark/overnight_run.log"
mkdir -p data/benchmark/results

exec > >(tee -a "$LOG") 2>&1
echo ""
echo "============================================"
echo "  Cloud-Gym Overnight Evaluation"
echo "  Started: $(date)"
echo "============================================"

# Phase 1: Baselines (llama3.2:3b)
echo ""
echo "=== [$(date +%H:%M:%S)] Baseline: llama3.2:3b ==="
python scripts/run_baselines.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --models "llama3.2:3b" \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] llama3.2:3b done ==="

# Phase 2: Baselines (qwen2.5-coder:7b)
echo ""
echo "=== [$(date +%H:%M:%S)] Baseline: qwen2.5-coder:7b ==="
python scripts/run_baselines.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --models "qwen2.5-coder:7b" \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] qwen2.5-coder:7b done ==="

# Phase 3: Baselines (qwen2.5-coder:32b)
echo ""
echo "=== [$(date +%H:%M:%S)] Baseline: qwen2.5-coder:32b ==="
python scripts/run_baselines.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --models "qwen2.5-coder:32b" \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] qwen2.5-coder:32b done ==="

# Phase 4: Fine-tuned model evaluation
echo ""
echo "=== [$(date +%H:%M:%S)] Fine-tuned model evaluation ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --skip-baselines \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] Fine-tuned eval done ==="

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Results in: data/benchmark/results/"
ls -la data/benchmark/results/*.json 2>/dev/null
