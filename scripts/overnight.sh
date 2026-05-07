#!/usr/bin/env bash
# Overnight pipeline: forward synthesis → format → fine-tune.
# Reverse-description is assumed already complete (data/taxi/reverse_pairs.jsonl).
#
# Run: nohup ./scripts/overnight.sh > /tmp/overnight.log 2>&1 &
# Each step has its own log under /tmp/overnight-<step>.log; the master log
# /tmp/overnight.log records start/exit of each step. Steps run sequentially
# because they all need the GPU; failures are recorded but don't kill the chain.

set -u
cd "$(dirname "$0")/.."

LOGD=/tmp/overnight-$(date +%Y%m%d-%H%M)
mkdir -p "$LOGD"
MASTER="$LOGD/all.log"

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"
}

step() {
  local name="$1"; shift
  log "=== START: $name ==="
  local sl="$LOGD/$name.log"
  ( time "$@" ) > "$sl" 2>&1
  local rc=$?
  log "=== EXIT $rc: $name (log: $sl, lines=$(wc -l < "$sl"))"
  return $rc
}

export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH=/Users/jhammant/dev/Cloud-Gym

log "overnight pipeline starting; logs in $LOGD"
log "  reverse pairs already on disk: $(wc -l < data/taxi/reverse_pairs.jsonl)"

# ---- Step 1: forward synthesis ----------------------------------------------
# 5000 target. Each call ~5-7s on lmstudio coder-next; ~7-10 hrs wall.
# Resumable — if it crashes, re-running picks up at the last checkpoint.
step forward python3 scripts/generate_taxi_pairs.py --mode forward --limit 5000

forward_pairs=$(wc -l < data/taxi/forward_pairs.jsonl 2>/dev/null || echo 0)
log "forward complete: $forward_pairs pairs total"

# ---- Step 2: format ----------------------------------------------------------
step format python3 scripts/format_taxi_finetuning.py

if [ -f data/finetune-taxi/train.jsonl ]; then
  train_n=$(wc -l < data/finetune-taxi/train.jsonl)
  log "format complete: $train_n training pairs"
else
  log "format failed — no training data; SKIPPING fine-tune step"
  log "overnight pipeline DONE (skipped fine-tune)"
  exit 1
fi

# ---- Step 3: fine-tune -------------------------------------------------------
# ~2 hours on M4. Adapter lands at data/models/taxi-nl-adapter/.
step finetune bash scripts/finetune_taxi.sh

if [ -d data/models/taxi-nl-adapter ]; then
  log "fine-tune complete: adapter at data/models/taxi-nl-adapter"
else
  log "fine-tune did not produce an adapter directory; check $LOGD/finetune.log"
fi

log "overnight pipeline DONE"
log "summary:"
log "  forward pairs:    $(wc -l < data/taxi/forward_pairs.jsonl 2>/dev/null || echo '?')"
log "  formatted train:  $(wc -l < data/finetune-taxi/train.jsonl 2>/dev/null || echo '?')"
log "  formatted valid:  $(wc -l < data/finetune-taxi/valid.jsonl 2>/dev/null || echo '?')"
log "  adapter dir:      $([ -d data/models/taxi-nl-adapter ] && echo present || echo missing)"
log "  master log:       $MASTER"
