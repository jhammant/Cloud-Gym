#!/bin/bash
# Cloud-Gym v2: Steps 5-6 (32B fine-tune + eval)
# Run after Steps 1-4 complete
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="data/benchmark/v2_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

# Step 5: Fine-tune 32B
echo ""
echo "=== [$(date +%H:%M:%S)] Step 5: Fine-tune 32B model ==="
python scripts/train.py \
    --model mlx-community/Qwen2.5-Coder-32B-Instruct-4bit \
    --adapter-path data/models/iac-repair-32b-adapter \
    --iters 600
echo "=== [$(date +%H:%M:%S)] 32B training done ==="

# Step 6: Eval 32B fine-tune (uses new concurrent evaluator)
echo ""
echo "=== [$(date +%H:%M:%S)] Step 6: Eval 32B fine-tuned ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results_v2 \
    --adapter-path data/models/iac-repair-32b-adapter \
    --base-model mlx-community/Qwen2.5-Coder-32B-Instruct-4bit \
    --skip-baselines \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] 32B eval done ==="

echo ""
echo "============================================"
echo "  Steps 5-6 complete! $(date)"
echo "============================================"
echo ""
echo "Results:"
ls -la data/benchmark/results_v2/*.json 2>/dev/null
