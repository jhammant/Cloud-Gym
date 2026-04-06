#!/bin/bash
# Cloud-Gym: Wait for 7B training, then run eval + Gemma 4 baseline
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate

LOG="data/models/iac-repair-7b-adapter/pipeline.log"

# Wait for training process to finish
TRAIN_PID=$(pgrep -f "run_7b_train.sh" | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "[$(date +%H:%M:%S)] Waiting for 7B training (PID $TRAIN_PID) to complete..." | tee -a "$LOG"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 30
    done
fi

# Verify training actually produced a final adapter
if [ ! -f "data/models/iac-repair-7b-adapter/adapters.safetensors" ]; then
    echo "[$(date +%H:%M:%S)] ERROR: No final adapter found. Training may have failed." | tee -a "$LOG"
    exit 1
fi

# Check it's not just a single early checkpoint (the bug we hit before)
CKPT_COUNT=$(ls data/models/iac-repair-7b-adapter/0*_adapters.safetensors 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] Training done. Found $CKPT_COUNT checkpoints." | tee -a "$LOG"

# Phase 1: Evaluate 7B fine-tuned model
echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  7B Fine-tuned Eval" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

python scripts/evaluate.py \
    --skip-baselines \
    --adapter-path data/models/iac-repair-7b-adapter \
    --base-model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
    --n-attempts 3 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] 7B eval done." | tee -a "$LOG"

# Phase 2: Evaluate Gemma 4 26B via LM Studio
echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  Gemma 4 26B-a4b Eval (LM Studio)" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# Check LM Studio is running
if curl -s http://localhost:1234/api/v1/chat -H "Content-Type: application/json" \
    -d '{"model":"google/gemma-4-26b-a4b","system_prompt":"test","input":"hi"}' > /dev/null 2>&1; then
    python scripts/evaluate.py \
        --skip-baselines \
        --skip-finetuned \
        --lmstudio-models "google/gemma-4-26b-a4b" \
        --n-attempts 3 2>&1 | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] Gemma 4 eval done." | tee -a "$LOG"
else
    echo "[$(date +%H:%M:%S)] WARNING: LM Studio not running, skipping Gemma 4 eval." | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  All evaluations complete: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Results:" | tee -a "$LOG"
ls -la data/benchmark/results/*.json | tee -a "$LOG"
