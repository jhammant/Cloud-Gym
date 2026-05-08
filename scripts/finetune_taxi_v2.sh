#!/usr/bin/env bash
# Fine-tune Qwen2.5-Coder-3B on NL->Taxi data with LoRA.
# Run: ./scripts/finetune_taxi.sh
# Same hyperparameters as the IaC repair finetune (proven to converge on M4).

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Fine-tuning Qwen2.5-Coder-3B-Instruct (4-bit) on NL->Taxi with LoRA ==="
echo "Data: data/finetune-taxi/{train,valid,test}.jsonl"
echo "Output: data/models/taxi-nl-adapter-v2/"
echo ""

TRAIN_LINES=$(wc -l < data/finetune-taxi/train.jsonl)
echo "train pairs: $TRAIN_LINES"
echo ""

.venv/bin/python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --train \
  --data data/finetune-taxi \
  --batch-size 1 \
  --iters 2000 \
  --learning-rate 2e-5 \
  --num-layers 16 \
  --max-seq-length 2048 \
  --adapter-path data/models/taxi-nl-adapter-v2 \
  --steps-per-report 100 \
  --steps-per-eval 250 \
  --val-batches 10 \
  --save-every 500

echo ""
echo "=== Training complete! Adapter -> data/models/taxi-nl-adapter-v2/ ==="
