#!/usr/bin/env bash
# Fine-tune Qwen2.5-Coder-3B on IaC repair data with LoRA
# Run: ./scripts/finetune.sh
# Takes ~2 hours on M4 24GB. Peak memory ~6GB.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Fine-tuning Qwen2.5-Coder-3B-Instruct (4-bit) with LoRA ==="
echo "Data: data/finetune/{train,valid,test}.jsonl"
echo "Output: data/models/iac-repair-adapter/"
echo ""

.venv/bin/python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --train \
  --data data/finetune \
  --batch-size 1 \
  --iters 600 \
  --learning-rate 2e-5 \
  --num-layers 8 \
  --max-seq-length 2048 \
  --adapter-path data/models/iac-repair-adapter \
  --steps-per-report 50 \
  --steps-per-eval 200 \
  --val-batches 10 \
  --save-every 200

echo ""
echo "=== Training complete! ==="
echo "Adapter saved to: data/models/iac-repair-adapter/"
echo ""
echo "Next steps:"
echo "  1. Fuse adapter:  ./scripts/export_model.sh"
echo "  2. Evaluate:      .venv/bin/python3 scripts/run_baselines.py --models cloudgym-repair,qwen2.5-coder:7b --n-attempts 3"
