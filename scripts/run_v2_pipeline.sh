#!/bin/bash
# Cloud-Gym v2: Improved eval + 7B fine-tune + 32b baseline
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="data/benchmark/v2_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  Cloud-Gym v2 Pipeline"
echo "  Started: $(date)"
echo "============================================"

# Step 1: Eval improved 3B fine-tune (with temperature sampling)
echo ""
echo "=== [$(date +%H:%M:%S)] Step 1: Eval 3B fine-tuned (v2 data + temp=0.3) ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results_v2 \
    --skip-baselines \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] 3B eval done ==="

# Step 2: Fine-tune 7B
echo ""
echo "=== [$(date +%H:%M:%S)] Step 2: Fine-tune 7B model ==="
python scripts/train.py \
    --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --adapter-path data/models/iac-repair-7b-adapter \
    --iters 600
echo "=== [$(date +%H:%M:%S)] 7B training done ==="

# Step 3: Eval 7B fine-tune
echo ""
echo "=== [$(date +%H:%M:%S)] Step 3: Eval 7B fine-tuned ==="
python scripts/evaluate.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results_v2 \
    --adapter-path data/models/iac-repair-7b-adapter \
    --base-model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --skip-baselines \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] 7B eval done ==="

# Step 4: 32b baseline
echo ""
echo "=== [$(date +%H:%M:%S)] Step 4: Baseline qwen2.5-coder:32b ==="
python scripts/run_baselines.py \
    --benchmark data/benchmark/benchmark.jsonl \
    --output-dir data/benchmark/results_v2 \
    --models "qwen2.5-coder:32b" \
    --n-attempts 3
echo "=== [$(date +%H:%M:%S)] 32b baseline done ==="

# Step 5: Fine-tune 32B
echo ""
echo "=== [$(date +%H:%M:%S)] Step 5: Fine-tune 32B model ==="
python scripts/train.py \
    --model mlx-community/Qwen2.5-Coder-32B-Instruct-4bit \
    --adapter-path data/models/iac-repair-32b-adapter \
    --iters 600
echo "=== [$(date +%H:%M:%S)] 32B training done ==="

# Step 6: Eval 32B fine-tune
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
echo "  All done! $(date)"
echo "============================================"
echo ""
echo "Results:"
ls -la data/benchmark/results_v2/*.json 2>/dev/null
