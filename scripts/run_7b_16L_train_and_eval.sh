#!/bin/bash
# Cloud-Gym: Train 7B with 16 LoRA layers, smoke test, then full eval
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate

ADAPTER="data/models/iac-repair-7b-adapter-16L"
LOG="$ADAPTER/train_and_eval.log"

echo "============================================" | tee "$LOG"
echo "  7B 16-Layer LoRA: Train + Eval" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# Step 1: Train
echo "" | tee -a "$LOG"
echo "=== Training (600 iters, 16 layers) ===" | tee -a "$LOG"

python scripts/train.py \
    --model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
    --adapter-path "$ADAPTER" \
    --iters 600 2>&1 | tee -a "$LOG"

if [ $? -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] ERROR: Training failed!" | tee -a "$LOG"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Training complete." | tee -a "$LOG"

# Step 2: Smoke test (20 entries, 1 attempt)
echo "" | tee -a "$LOG"
echo "=== Smoke test: 20 entries ===" | tee -a "$LOG"

python3 << PYEOF 2>&1 | tee -a "$LOG"
import json, asyncio, tempfile
from pathlib import Path
from cloudgym.benchmark.evaluator import Evaluator
from scripts.evaluate import make_mlx_repair_fn

with open('data/benchmark/benchmark.jsonl') as f:
    entries = [json.loads(line) for line in f][:20]

mini_path = Path(tempfile.mktemp(suffix='.jsonl'))
with open(mini_path, 'w') as f:
    for e in entries:
        f.write(json.dumps(e) + '\n')

evaluator = Evaluator(str(mini_path))
repair_fn = make_mlx_repair_fn(
    'mlx-community/Qwen2.5-Coder-7B-Instruct-4bit',
    '$ADAPTER',
)
report = asyncio.run(evaluator.evaluate_model(
    model_fn=repair_fn,
    model_name='7b-16L-smoke',
    n_attempts=1,
    k_values=[1],
))

pass1 = report.pass_at_k.get(1, 0)
print(f"\nSmoke test pass@1: {pass1:.3f}")
for fmt, metrics in report.per_format.items():
    print(f"  {fmt}: {metrics.get(1, 0):.3f}")

mini_path.unlink(missing_ok=True)

with open('/tmp/7b_16L_smoke.txt', 'w') as f:
    f.write(f"{pass1:.4f}")
PYEOF

# Step 3: Check smoke test and run full eval if >50%
SMOKE=$(cat /tmp/7b_16L_smoke.txt 2>/dev/null || echo "0.0")
echo "" | tee -a "$LOG"
echo "Smoke result: $SMOKE" | tee -a "$LOG"

if python3 -c "exit(0 if float('$SMOKE') > 0.5 else 1)"; then
    echo "=== PASSED — running full 188-entry eval ===" | tee -a "$LOG"

    python scripts/evaluate.py \
        --skip-baselines \
        --adapter-path "$ADAPTER" \
        --base-model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
        --n-attempts 3 2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "Full eval complete: $(date)" | tee -a "$LOG"
else
    echo "=== FAILED smoke test ($SMOKE < 0.50) — skipping full run ===" | tee -a "$LOG"
    exit 1
fi
