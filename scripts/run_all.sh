#!/bin/bash
# Cloud-Gym: Full batch pipeline
# Generates data, builds benchmark, trains model, runs evaluations
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo "  Cloud-Gym Full Pipeline"
echo "============================================"

# Phase 1: Install dependencies
echo ""
echo "=== Phase 1: Install dependencies ==="
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev,mlx]"
fi

# Phase 2: Generate gold configs (parallel)
echo ""
echo "=== Phase 2: Generate gold configs ==="
echo "  Terraform (500 configs) and CloudFormation (200 configs) in parallel..."
python scripts/generate_gold.py -n 500 -s 42 &
PID_TF=$!
python scripts/generate_gold_cf.py -n 200 -s 1000 &
PID_CF=$!
wait $PID_TF
echo "  Terraform gold configs done."
wait $PID_CF
echo "  CloudFormation gold configs done."

# Phase 3: Generate training data
echo ""
echo "=== Phase 3: Generate training data ==="
python scripts/generate.py --programmatic-variants 8 --skip-agentic

# Phase 3b: Build benchmark
echo ""
echo "=== Phase 3b: Build benchmark ==="
python scripts/build_benchmark.py --target-size 200

# Phase 3c: Format for fine-tuning
echo ""
echo "=== Phase 3c: Format for fine-tuning ==="
python scripts/format_for_finetuning.py

# Phase 4a: Fine-tune model
echo ""
echo "=== Phase 4a: Fine-tune model (MLX LoRA) ==="
python scripts/train.py --iters 1000

# Phase 4b: Run baselines
echo ""
echo "=== Phase 4b: Run Ollama baselines ==="
python scripts/run_baselines.py \
    --models "llama3.2:3b,qwen2.5-coder:7b,qwen2.5-coder:32b" \
    --n-attempts 5

# Phase 5: Evaluate fine-tuned model
echo ""
echo "=== Phase 5: Evaluate fine-tuned model ==="
python scripts/evaluate.py --skip-baselines --n-attempts 5

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo ""
echo "Results:"
echo "  Training data: data/training/"
echo "  Benchmark:     data/benchmark/benchmark.jsonl"
echo "  Fine-tune:     data/models/iac-repair-adapter/"
echo "  Results:       data/benchmark/results/"
