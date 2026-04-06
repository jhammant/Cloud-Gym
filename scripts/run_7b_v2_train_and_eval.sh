#!/bin/bash
# Cloud-Gym: Retrain 7B with improved hyperparameters
# Fixes: more LoRA layers (20/28), longer seq length (4096), lower scale (10.0), more iters (1000)
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="data/models/7b_v2_train_and_eval.log"
exec > >(tee "$LOG") 2>&1

echo "============================================"
echo "  7B v2 Training + Eval"
echo "  Started: $(date)"
echo "============================================"

echo ""
echo "=== [$(date +%H:%M:%S)] Training 7B v2 (20 layers, seq 4096, scale 10.0, 1000 iters) ==="
python scripts/train.py \
    --model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
    --adapter-path "data/models/iac-repair-7b-adapter-v2" \
    --iters 1000

echo ""
echo "=== [$(date +%H:%M:%S)] Training done. Running eval... ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results \
    --skip-baselines \
    --adapter-path "data/models/iac-repair-7b-adapter-v2" \
    --base-model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
    --n-attempts 3

echo ""
echo "=== [$(date +%H:%M:%S)] All done. ==="
