"""P4: synthetic (NL → Taxi) training pair generation.

Two modes, both write resumably to data/taxi/{reverse,forward}_pairs.jsonl:

  --mode reverse
    For every snippet in valid_corpus.jsonl, ask the LLM to write 3 NL
    descriptions in distinct styles (terse / task / doc). Pair each with
    the snippet's already-validated Taxi as the gold answer. No validator
    gating needed — gold is pre-validated.

  --mode forward
    For each (domain × construct) bucket, ask the LLM to emit a
    {"description","taxi"} JSON pair. Run the produced Taxi through the
    strict validator. If invalid, feed the compiler error back for ONE
    self-correct attempt. Drop if still invalid.

Resumability: each mode tracks completed snippet/bucket indices in a
sidecar `<mode>_progress.json`. Re-running the script picks up where it
left off, so a crash or interrupt loses at most one in-flight call.

Default model adapter: lmstudio-api:qwen/qwen3-coder-next (already loaded
in your LMStudio for other work; we share, no model swap triggered).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# reuse adapters from eval_taxi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_taxi import (  # type: ignore
    LMStudioRespAPIAdapter,
    OllamaAdapter,
    OpenAICompatAdapter,
    ModelOutput,
    make_adapter,
    _strip_fences,
)

from cloudgym.taxi.validator import TaxiValidator

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data/taxi/valid_corpus.jsonl"
OUT_DIR = REPO / "data/taxi"


# --------------------------------------------------------------------- prompts
SYSTEM_REVERSE = """You translate Taxi schema code into the kind of plain-English description a developer writes when asking another developer to build it.

Taxi is the schema language used by Orbital (orbitalhq.com). For each Taxi snippet, produce exactly 3 descriptions in 3 different styles, separated by lines containing only '---'. The styles are:

TERSE: one short sentence in the imperative voice ("Define a Customer model with id and email")
TASK:  task-card phrasing ("As a developer, I want a Customer model that has an id and an email field")
DOC:   doc-comment phrasing ("Customer represents a registered buyer with a unique identifier and a contact email")

Refer to the schema by what it MEANS in plain English (a Customer, an email address, a price), not by Taxi keywords. Do NOT mention 'inherits', 'model', 'type', or other Taxi keywords. Output ONLY the 3 descriptions separated by --- lines, no other prose."""

SYSTEM_FORWARD = """You generate (description, Taxi schema) training pairs for fine-tuning. Taxi is the schema language used by Orbital (orbitalhq.com).

Given a domain and a construct bucket, emit ONE JSON object with two keys:
  "description": a plain-English request a developer might write
  "taxi":        the corresponding Taxi code that satisfies the request

Taxi syntax reminders (these are STRICT — the compiler rejects deviations):
- `type X inherits Y` for primitive aliases (Y ∈ String, Int, Decimal, Boolean, Instant, Date)
  ✓ `type CustomerId inherits String`
  ✗ `type CustomerId = string`        (TypeScript syntax — invalid Taxi)
  ✗ `type CustomerId : String`        (no colon)
- `model X { field : Type }` for record-shaped data
  ✓ `model Customer { id : CustomerId }`
  ✗ `Customer {}`                     (model keyword required)
- `enum X { A, B, C }` — closed sets, NEVER inherit
  ✓ `enum Status { ACTIVE, INACTIVE }`
- `service X { operation foo(arg : T) : R }`
  ✓ `service C { operation get(id: CustomerId): Customer }`
- `annotation X` or `annotation X { field : T }`, applied via `@X` or `@X(field=...)`
- `T?` nullable on the TYPE side (`field : String?`), NEVER on the field name (`field?: T` is wrong)
- `T[]` for arrays
- `find { Foo[] }` for TaxiQL queries; `find { Foo } as { name : T }` for projections

Three reference examples — match this style and structure:

Example 1 (model_simple):
```json
{"description": "Define a Customer model with an id (CustomerId, a string-derived type) and an email (EmailAddress).", "taxi": "type CustomerId inherits String\\ntype EmailAddress inherits String\\nmodel Customer {\\n  id : CustomerId\\n  email : EmailAddress\\n}"}
```

Example 2 (service_basic):
```json
{"description": "Create an OrderService with an operation findOrderById that takes an OrderId string and returns a String.", "taxi": "type OrderId inherits String\\nservice OrderService {\\n  operation findOrderById(id : OrderId) : String\\n}"}
```

Example 3 (enum + model):
```json
{"description": "Define an OrderStatus enum (PENDING, PAID, SHIPPED) and an Order model with an id and a status.", "taxi": "type OrderId inherits String\\nenum OrderStatus { PENDING, PAID, SHIPPED }\\nmodel Order {\\n  id : OrderId\\n  status : OrderStatus\\n}"}
```

Output JSON only, no prose or markdown fences."""


REVERSE_USER_TPL = "Taxi:\n```taxi\n{taxi}\n```\n\nProduce 3 descriptions in TERSE / TASK / DOC styles, separated by --- lines."

FORWARD_USER_TPL = (
    "Domain: {domain}\n"
    "Construct bucket: {bucket}\n"
    "Theme: {theme}\n\n"
    "Emit one JSON object: {{\"description\": \"...\", \"taxi\": \"...\"}}"
)

FORWARD_RETRY_TPL = (
    "Your previous output failed to compile. The compiler reported:\n"
    "{errors}\n\n"
    "Original Taxi:\n```taxi\n{taxi}\n```\n\n"
    "Emit a corrected JSON object: {{\"description\": \"...\", \"taxi\": \"...\"}}. "
    "Keep the description but fix the Taxi to compile."
)


# --------------------------------------------------------------------- reverse
@dataclass
class ReverseStats:
    snippets_attempted: int = 0
    snippets_with_3: int = 0
    pairs_emitted: int = 0
    parse_failures: int = 0
    model_errors: int = 0
    total_seconds: float = 0.0


def _parse_reverse_output(text: str) -> list[tuple[str, str]]:
    """Return list of (style, description) parsed from a reverse-mode LLM response."""
    if not text:
        return []
    text = _strip_fences(text)
    # split on lines that are exactly --- (with optional whitespace)
    parts = re.split(r"^\s*-{3,}\s*$", text, flags=re.MULTILINE)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"^\s*(TERSE|TASK|DOC)\s*:\s*(.+)$", p, re.IGNORECASE | re.DOTALL)
        if m:
            out.append((m.group(1).upper(), re.sub(r"\s+", " ", m.group(2).strip())))
        else:
            # accept un-prefixed as UNKNOWN
            out.append(("UNKNOWN", re.sub(r"\s+", " ", p)))
    return out


def reverse_run(
    adapter,
    corpus: list[dict],
    out_path: Path,
    progress_path: Path,
    limit: int | None = None,
    skip_construct: set[str] | None = None,
) -> ReverseStats:
    # load progress
    done_ids: set[str] = set()
    if progress_path.exists():
        done_ids = set(json.loads(progress_path.read_text()).get("done", []))
    print(f"reverse: {len(done_ids)} snippets already done")

    stats = ReverseStats()
    work: list[dict] = []
    for r in corpus:
        sid = _snippet_id(r)
        if sid in done_ids:
            continue
        if skip_construct and r.get("construct") in skip_construct:
            continue
        work.append(r)
    if limit is not None:
        work = work[:limit]
    print(f"reverse: {len(work)} snippets to process")

    out_f = out_path.open("a")
    t0 = time.time()
    for i, snippet in enumerate(work):
        sid = _snippet_id(snippet)
        stats.snippets_attempted += 1
        user = REVERSE_USER_TPL.format(taxi=snippet["taxi"].rstrip())
        out: ModelOutput = adapter.generate(SYSTEM_REVERSE, user, max_tokens=600)
        stats.total_seconds += out.elapsed_s
        if out.error:
            stats.model_errors += 1
            continue
        descriptions = _parse_reverse_output(out.text)
        if not descriptions:
            stats.parse_failures += 1
            continue
        if len([d for d in descriptions if d[1]]) >= 3:
            stats.snippets_with_3 += 1
        for style, desc in descriptions:
            if not desc:
                continue
            rec = {
                "snippet_id": sid,
                "construct": snippet.get("construct"),
                "source_file": snippet.get("source_file"),
                "style": style,
                "description": desc,
                "taxi": snippet["taxi"],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.pairs_emitted += 1
        out_f.flush()
        done_ids.add(sid)
        # checkpoint every 5
        if i % 5 == 0 or i + 1 == len(work):
            progress_path.write_text(json.dumps({"done": sorted(done_ids)}))
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(work) - i - 1) / max(0.01, rate)
            print(f"  [{i+1}/{len(work)}] pairs={stats.pairs_emitted} "
                  f"errors={stats.model_errors+stats.parse_failures} "
                  f"avg={stats.total_seconds/max(1,stats.snippets_attempted):.1f}s/snippet "
                  f"ETA {eta/60:.1f}min")
    out_f.close()
    progress_path.write_text(json.dumps({"done": sorted(done_ids)}))
    return stats


# --------------------------------------------------------------------- forward
DOMAINS = [
    "trading", "banking", "healthcare", "rideshare", "ecommerce",
    "iot", "logistics", "social", "content", "telecom",
    "energy", "education", "real_estate", "media", "supply_chain",
    "fintech", "gaming", "travel", "hr", "agriculture",
]
CONSTRUCT_BUCKETS = [
    ("type", "primitive-derived alias types only (3-5)"),
    ("model_simple", "single model with 3-6 fields and supporting type aliases"),
    ("model_nested", "two related models, one referencing the other"),
    ("enum", "a closed enum and a model that uses it"),
    ("service_basic", "a service with 1-2 simple operations"),
    ("service_crud", "a service with create/read/update/delete style operations"),
    ("annotation", "an annotation definition and its application to a model"),
    ("query_basic", "a small schema followed by a TaxiQL find query"),
    ("query_projection", "a schema followed by a find query with `as { ... }` projection"),
    ("multi_block", "3-5 declaration blocks that compose a small subdomain"),
]
THEMES = [
    "core entity records", "audit logging", "billing or payments",
    "inventory or stock levels", "user authentication", "notifications",
    "reporting and aggregations", "external integrations", "lifecycle state",
    "search and filtering",
]


@dataclass
class ForwardStats:
    attempted: int = 0
    valid_first_try: int = 0
    valid_after_retry: int = 0
    invalid_dropped: int = 0
    parse_failures: int = 0
    model_errors: int = 0
    total_seconds: float = 0.0


def _extract_json_object(text: str) -> dict | None:
    """Pull a single JSON object from the LLM output, tolerating prose."""
    if not text:
        return None
    text = _strip_fences(text)
    # try direct first
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first { ... } that parses
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str: str | None = None
        for i in range(start, len(text)):
            c = text[i]
            if in_str is not None:
                if c == "\\":
                    continue
                if c == in_str:
                    in_str = None
            elif c == '"':
                in_str = '"'
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def forward_run(
    adapter,
    validator: TaxiValidator,
    out_path: Path,
    progress_path: Path,
    n_pairs: int,
    seed: int = 7,
) -> ForwardStats:
    # progress: count of pairs emitted
    emitted = 0
    if progress_path.exists():
        emitted = json.loads(progress_path.read_text()).get("emitted", 0)
    print(f"forward: {emitted} pairs already done, target {n_pairs}")

    rng = random.Random(seed)
    stats = ForwardStats()
    out_f = out_path.open("a")
    t0 = time.time()

    while emitted < n_pairs:
        domain = rng.choice(DOMAINS)
        bucket_name, bucket_desc = rng.choice(CONSTRUCT_BUCKETS)
        theme = rng.choice(THEMES)
        user = FORWARD_USER_TPL.format(domain=domain, bucket=f"{bucket_name} — {bucket_desc}", theme=theme)
        stats.attempted += 1
        out: ModelOutput = adapter.generate(SYSTEM_FORWARD, user, max_tokens=900)
        stats.total_seconds += out.elapsed_s
        if out.error:
            stats.model_errors += 1
            continue
        obj = _extract_json_object(out.text)
        if not obj or "description" not in obj or "taxi" not in obj:
            stats.parse_failures += 1
            continue
        candidate_taxi = obj["taxi"]
        try:
            res = validator.validate(candidate_taxi, source_name="forward.taxi")
        except Exception:
            stats.model_errors += 1
            continue
        passed = res.is_valid
        used_retry = False
        if not passed:
            # one self-correct attempt
            err_text = "\n".join(f"L{e.line}C{e.char}: {e.detailMessage}" for e in res.errors[:5])
            retry_user = FORWARD_RETRY_TPL.format(errors=err_text, taxi=candidate_taxi)
            out2 = adapter.generate(SYSTEM_FORWARD, retry_user, max_tokens=900)
            stats.total_seconds += out2.elapsed_s
            if out2.error:
                stats.invalid_dropped += 1
                continue
            obj2 = _extract_json_object(out2.text)
            if obj2 and "description" in obj2 and "taxi" in obj2:
                try:
                    res2 = validator.validate(obj2["taxi"], source_name="forward_retry.taxi")
                    if res2.is_valid:
                        obj = obj2
                        candidate_taxi = obj2["taxi"]
                        passed = True
                        used_retry = True
                        stats.valid_after_retry += 1
                except Exception:
                    pass
        if not passed:
            stats.invalid_dropped += 1
            continue
        if not used_retry:
            stats.valid_first_try += 1
        rec = {
            "domain": domain,
            "bucket": bucket_name,
            "theme": theme,
            "description": obj["description"],
            "taxi": candidate_taxi,
            "used_retry": used_retry,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        emitted += 1
        # checkpoint every 10
        if emitted % 10 == 0:
            progress_path.write_text(json.dumps({"emitted": emitted}))
        if stats.attempted % 25 == 0:
            elapsed = time.time() - t0
            rate = stats.attempted / elapsed
            eta_s = (n_pairs - emitted) / max(0.01, rate * (emitted / max(1, stats.attempted)))
            print(f"  emitted={emitted}/{n_pairs}  attempted={stats.attempted}  "
                  f"first-try={stats.valid_first_try}  retry={stats.valid_after_retry}  "
                  f"dropped={stats.invalid_dropped}  ETA {eta_s/60:.1f}min")
    progress_path.write_text(json.dumps({"emitted": emitted}))
    out_f.close()
    return stats


# --------------------------------------------------------------------- helpers
def _snippet_id(rec: dict) -> str:
    """Stable id for a corpus snippet: source_file:source_line if present, else hash."""
    sf = rec.get("source_file")
    sl = rec.get("source_line")
    if sf and sl:
        return f"{sf}:{sl}"
    import hashlib
    return hashlib.sha1(rec["taxi"].encode()).hexdigest()[:12]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reverse", "forward"], required=True)
    p.add_argument("--model", default="lmstudio-api:qwen/qwen3-coder-next",
                   help="adapter spec, see scripts/eval_taxi.py")
    p.add_argument("--limit", type=int, default=None,
                   help="reverse: max snippets; forward: target pair count")
    p.add_argument("--probe", action="store_true",
                   help="probe mode: 20 snippets/pairs only, write to *_probe.jsonl")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    adapter = make_adapter(args.model)
    suffix = "_probe" if args.probe else ""
    if args.mode == "reverse":
        out_path = OUT_DIR / f"reverse_pairs{suffix}.jsonl"
        progress_path = OUT_DIR / f"reverse_progress{suffix}.json"
        corpus = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
        print(f"reverse: corpus={len(corpus)} snippets, model={adapter.name}, output={out_path.name}")
        limit = 20 if args.probe else args.limit
        stats = reverse_run(adapter, corpus, out_path, progress_path, limit=limit)
        print(f"\n=== reverse summary ===")
        print(json.dumps(stats.__dict__, indent=2))
    else:
        out_path = OUT_DIR / f"forward_pairs{suffix}.jsonl"
        progress_path = OUT_DIR / f"forward_progress{suffix}.json"
        n = 20 if args.probe else (args.limit or 5000)
        print(f"forward: target={n} pairs, model={adapter.name}, output={out_path.name}")
        with TaxiValidator() as v:
            stats = forward_run(adapter, v, out_path, progress_path, n_pairs=n, seed=args.seed)
        print(f"\n=== forward summary ===")
        print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
