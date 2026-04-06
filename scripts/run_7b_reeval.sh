#!/bin/bash
# Cloud-Gym: Re-evaluate 7B with stop token fix
# Runs 20-entry smoke test first, then full run if >50% pass
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source .venv/bin/activate

LOG="data/models/iac-repair-7b-adapter/reeval.log"

echo "============================================" | tee "$LOG"
echo "  7B Re-evaluation (stop token fix)" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# Step 1: Quick smoke test on 20 entries
echo "" | tee -a "$LOG"
echo "=== Smoke test: 20 entries ===" | tee -a "$LOG"

python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, asyncio, tempfile
from pathlib import Path
from scripts.evaluate import make_mlx_repair_fn, _strip_markdown_fences, REPAIR_SYSTEM_PROMPT
from cloudgym.benchmark.evaluator import Evaluator

# Build a mini benchmark with first 20 entries
with open('data/benchmark/benchmark.jsonl') as f:
    entries = [json.loads(line) for line in f][:20]

mini_path = Path(tempfile.mktemp(suffix='.jsonl'))
with open(mini_path, 'w') as f:
    for e in entries:
        f.write(json.dumps(e) + '\n')

evaluator = Evaluator(str(mini_path))
print(f"Mini benchmark: {len(evaluator.dataset)} entries")

repair_fn = make_mlx_repair_fn(
    'mlx-community/Qwen2.5-Coder-7B-Instruct-4bit',
    'data/models/iac-repair-7b-adapter',
)

report = asyncio.run(evaluator.evaluate_model(
    model_fn=repair_fn,
    model_name='finetuned:iac-repair-7b-adapter',
    n_attempts=1,
    k_values=[1],
))

pass1 = report.pass_at_k.get(1, 0)
print(f"\nSmoke test pass@1: {pass1:.3f} ({int(pass1 * len(evaluator.dataset))}/{len(evaluator.dataset)})")

mini_path.unlink(missing_ok=True)

# Write result for bash to read
with open('/tmp/7b_smoke_result.txt', 'w') as f:
    f.write(f"{pass1:.4f}")
PYEOF

# Step 2: Check smoke test result
SMOKE_RESULT=$(cat /tmp/7b_smoke_result.txt 2>/dev/null || echo "0.0")
echo "" | tee -a "$LOG"
echo "Smoke test result: $SMOKE_RESULT" | tee -a "$LOG"

# If >50% pass, run full eval
if python3 -c "exit(0 if float('$SMOKE_RESULT') > 0.5 else 1)"; then
    echo "" | tee -a "$LOG"
    echo "=== PASSED smoke test, running full 188-entry eval ===" | tee -a "$LOG"

    python scripts/evaluate.py \
        --skip-baselines \
        --adapter-path data/models/iac-repair-7b-adapter \
        --base-model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" \
        --n-attempts 3 2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "Full eval complete: $(date)" | tee -a "$LOG"
else
    echo "" | tee -a "$LOG"
    echo "=== FAILED smoke test ($SMOKE_RESULT < 0.50), skipping full run ===" | tee -a "$LOG"
    exit 1
fi
