#!/bin/bash
# Cloud-Gym: Wait for 7b fine-tuned eval, then run Gemma 4 26B baseline
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate

# Wait for the 7b fine-tuned eval to finish
echo "Waiting for 7b fine-tuned eval (PID 50857) to complete..."
while kill -0 50857 2>/dev/null; do
    sleep 30
done
echo "7b eval finished at $(date)"

# Run Gemma 4 26B via LM Studio
echo ""
echo "============================================"
echo "  Gemma 4 26B-a4b Evaluation"
echo "  Started: $(date)"
echo "============================================"

python scripts/evaluate.py \
    --skip-baselines \
    --skip-finetuned \
    --lmstudio-models "google/gemma-4-26b-a4b" \
    --n-attempts 3

echo ""
echo "============================================"
echo "  Gemma 4 eval complete: $(date)"
echo "============================================"
