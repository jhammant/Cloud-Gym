#!/bin/bash
# Cloud-Gym: Resilient 7B fine-tuning with auto-resume on crash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate

ADAPTER_DIR="data/models/iac-repair-7b-adapter"
LOG="$ADAPTER_DIR/train_v3.log"
MAX_RETRIES=3
RETRY=0

echo "============================================" | tee -a "$LOG"
echo "  7B Fine-tune (resilient mode)" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "  Grad checkpoint: ON" | tee -a "$LOG"
echo "  Save every: 50 iters" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

while [ $RETRY -lt $MAX_RETRIES ]; do
    # Find latest checkpoint to resume from
    LATEST_CKPT=$(ls -t "$ADAPTER_DIR"/0000*_adapters.safetensors 2>/dev/null | head -1)

    if [ -n "$LATEST_CKPT" ]; then
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] Resuming from checkpoint: $LATEST_CKPT (attempt $((RETRY+1))/$MAX_RETRIES)" | tee -a "$LOG"
        python scripts/train.py \
            --model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
            --adapter-path "$ADAPTER_DIR" \
            --iters 600 2>&1 | tee -a "$LOG"
    else
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] Starting fresh training (attempt $((RETRY+1))/$MAX_RETRIES)" | tee -a "$LOG"
        python scripts/train.py \
            --model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
            --adapter-path "$ADAPTER_DIR" \
            --iters 600 2>&1 | tee -a "$LOG"
    fi

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "" | tee -a "$LOG"
        echo "[$(date +%H:%M:%S)] Training completed successfully!" | tee -a "$LOG"
        break
    fi

    RETRY=$((RETRY + 1))
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] Training crashed (exit $EXIT_CODE). Retry $RETRY/$MAX_RETRIES in 10s..." | tee -a "$LOG"
    sleep 10
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "[$(date +%H:%M:%S)] FAILED after $MAX_RETRIES retries." | tee -a "$LOG"
    exit 1
fi

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  Training complete: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
