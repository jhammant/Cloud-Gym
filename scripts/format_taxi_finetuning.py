"""Format reverse + forward synthetic pairs into MLX chat JSONL for fine-tuning.

Reads:
  data/taxi/reverse_pairs.jsonl   {snippet_id, construct, style, description, taxi}
  data/taxi/forward_pairs.jsonl   {domain, bucket, theme, description, taxi, used_retry}

Excludes any pair whose Taxi gold matches a benchmark gold (so we don't leak
test items into training).

Writes MLX chat format to data/finetune-taxi/{train,valid,test}.jsonl
(80/10/10 stratified by source: reverse vs forward, plus by construct/bucket
where available).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAXI_DIR = REPO / "data/taxi"
OUT_DIR = REPO / "data/finetune-taxi"

SYSTEM_PROMPT = (
    "You translate natural-language requirements into idiomatic Taxi schema code. "
    "Taxi is the schema language used by Orbital (orbitalhq.com). "
    "Return ONLY the Taxi source. Do not include prose, explanation, or markdown fences."
)


def _content_hash(taxi: str) -> str:
    norm = "\n".join(line.rstrip() for line in taxi.replace("\r\n", "\n").splitlines())
    return hashlib.sha1(norm.encode()).hexdigest()


def _format_chat(description: str, taxi: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description.strip()},
            {"role": "assistant", "content": taxi.rstrip()},
        ]
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_path = TAXI_DIR / "benchmark.jsonl"
    bench_hashes: set[str] = set()
    if benchmark_path.exists():
        for line in benchmark_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            bench_hashes.add(_content_hash(r["gold_taxi"]))
            if r.get("in_context_schema"):
                bench_hashes.add(_content_hash(r["in_context_schema"] + "\n" + r["gold_taxi"]))
    print(f"benchmark gold hashes (excluded from training): {len(bench_hashes)}")

    pairs: list[dict] = []
    counts = Counter()

    rev_path = TAXI_DIR / "reverse_pairs.jsonl"
    if rev_path.exists():
        for line in rev_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            h = _content_hash(r["taxi"])
            if h in bench_hashes:
                counts["reverse_skipped_benchmark"] += 1
                continue
            pairs.append({
                "source": "reverse",
                "construct": r.get("construct", "?"),
                "style": r.get("style", "?"),
                "description": r["description"],
                "taxi": r["taxi"],
            })
            counts["reverse_kept"] += 1

    fwd_path = TAXI_DIR / "forward_pairs.jsonl"
    if fwd_path.exists():
        for line in fwd_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            h = _content_hash(r["taxi"])
            if h in bench_hashes:
                counts["forward_skipped_benchmark"] += 1
                continue
            pairs.append({
                "source": "forward",
                "construct": r.get("bucket", "?"),
                "domain": r.get("domain", "?"),
                "description": r["description"],
                "taxi": r["taxi"],
            })
            counts["forward_kept"] += 1

    if not pairs:
        raise SystemExit("no pairs found; run scripts/generate_taxi_pairs.py first")

    print(f"total kept: {len(pairs)}")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    # dedupe by (description, taxi) — same description+taxi shouldn't repeat
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for p in pairs:
        key = (p["description"][:200], p["taxi"][:200])
        if key in seen: continue
        seen.add(key)
        unique.append(p)
    print(f"deduped: {len(unique)} ({len(pairs) - len(unique)} dups)")

    # stratified shuffle: group by (source, construct) bucket then split 80/10/10
    rng = random.Random(11)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for p in unique:
        by_bucket[f"{p['source']}/{p['construct']}"].append(p)

    train, val, test = [], [], []
    for bucket, rows in by_bucket.items():
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        train.extend(rows[:n_train])
        val.extend(rows[n_train : n_train + n_val])
        test.extend(rows[n_train + n_val :])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")

    def _write(name: str, rows: list[dict]) -> None:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(_format_chat(r["description"], r["taxi"]), ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows)} -> {path.relative_to(REPO)}")

    _write("train", train)
    _write("valid", val)
    _write("test", test)

    # ALSO write a sidecar with raw pairs (for re-formatting later if SYSTEM_PROMPT changes)
    raw_path = OUT_DIR / "raw_pairs.jsonl"
    with raw_path.open("w") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  raw -> {raw_path.relative_to(REPO)}  ({len(unique)} records)")


if __name__ == "__main__":
    main()
