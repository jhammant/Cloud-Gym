"""Run NL→Taxi benchmark against a set of model adapters and compute metrics.

Adapters (added on demand):
  ollama       — http://localhost:11434  (e.g. qwen2.5-coder:7b, qwen2.5-coder:32b)
  lmstudio     — http://localhost:1234   (e.g. qwen3.6-27b-mlx)
  openrouter   — https://openrouter.ai/api/v1   (needs OPENROUTER_API_KEY)

Metrics:
  compile_pass_rate         — fraction whose output passes the strict validator
                              (multi-source if benchmark entry has in_context_schema)
  field_recall              — fraction of "name : Type" pairs in gold also present in output
  type_decl_recall          — fraction of top-level type/model/service/enum/annotation
                              declarations in gold also present in output

Outputs:
  data/taxi/eval_<model_id>.jsonl   per-entry model output + metrics
  data/taxi/eval_summary.md         pretty cross-model report

Use:
  python scripts/eval_taxi.py --models ollama:qwen2.5-coder:7b,lmstudio:qwen3.6-27b-mlx
  python scripts/eval_taxi.py --models ollama:qwen2.5-coder:7b --stuff   # context-stuffed variant
  python scripts/eval_taxi.py --list   # show available baselines
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib import request as _urlreq
from urllib.error import HTTPError, URLError

from cloudgym.taxi.validator import TaxiValidator, ValidationResult

REPO = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO / "data/taxi/benchmark.jsonl"
CORPUS_PATH = REPO / "data/taxi/valid_corpus.jsonl"
OUT_DIR = REPO / "data/taxi"

SYSTEM_PROMPT = (
    "You translate natural-language requirements into idiomatic Taxi schema code. "
    "Taxi is the schema language used by Orbital (orbitalhq.com). "
    "Return ONLY the Taxi source. Do not include prose, explanation, or markdown fences."
)


# ----------------------------------------------------------------------- helpers
def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped its answer."""
    text = text.strip()
    # ```taxi ... ``` or ``` ... ```
    m = re.match(r"^```(?:taxi|kotlin)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


# ----------------------------------------------------------------------- adapters
@dataclass
class ModelOutput:
    text: str
    elapsed_s: float
    error: str | None = None


class ModelAdapter:
    name: str

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> ModelOutput:
        raise NotImplementedError


class OllamaAdapter(ModelAdapter):
    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host
        self.name = f"ollama:{model}"

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> ModelOutput:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }).encode()
        req = _urlreq.Request(f"{self.host}/api/chat", data=body,
                              headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.time()
        try:
            with _urlreq.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read())
        except (URLError, HTTPError, TimeoutError) as e:
            return ModelOutput(text="", elapsed_s=time.time() - t0, error=f"{type(e).__name__}: {e}")
        return ModelOutput(text=payload.get("message", {}).get("content", ""),
                           elapsed_s=time.time() - t0)


class OpenAICompatAdapter(ModelAdapter):
    """For LMStudio / vLLM / any OpenAI-compatible server."""
    def __init__(self, model: str, host: str, api_key: str | None = None, label: str | None = None) -> None:
        self.model = model
        self.host = host
        self.api_key = api_key
        self.name = label or f"openai-compat:{model}"

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> ModelOutput:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = _urlreq.Request(f"{self.host}/chat/completions", data=body, headers=headers, method="POST")
        t0 = time.time()
        try:
            with _urlreq.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read())
        except (URLError, HTTPError, TimeoutError) as e:
            return ModelOutput(text="", elapsed_s=time.time() - t0, error=f"{type(e).__name__}: {e}")
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return ModelOutput(text="", elapsed_s=time.time() - t0, error=f"bad response: {e}")
        return ModelOutput(text=text or "", elapsed_s=time.time() - t0)


class LMStudioRespAPIAdapter(ModelAdapter):
    """LMStudio's /api/v1/chat 'responses' style endpoint.
    Uses system_prompt + input (single string) instead of messages[].
    Output shape: {"output": [{"type": "message", "content": "..."}], ...}.
    Useful for routing to whatever model the user has currently loaded
    without needing to control LMStudio model lifecycle."""
    def __init__(self, model: str, host: str = "http://localhost:1234") -> None:
        self.model = model
        self.host = host
        self.name = f"lmstudio-api:{model}"

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> ModelOutput:
        # LMStudio responses API uses `max_output_tokens`, not `max_tokens`.
        # `temperature` is accepted but ignored on some servers; we set it for parity.
        body = json.dumps({
            "model": self.model,
            "system_prompt": system,
            "input": user,
            "temperature": 0.0,
            "max_output_tokens": max_tokens,
        }).encode()
        req = _urlreq.Request(f"{self.host}/api/v1/chat", data=body,
                              headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.time()
        try:
            with _urlreq.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read())
        except (URLError, HTTPError, TimeoutError) as e:
            return ModelOutput(text="", elapsed_s=time.time() - t0, error=f"{type(e).__name__}: {e}")
        try:
            text = ""
            for item in payload.get("output", []):
                if item.get("type") == "message":
                    text += item.get("content", "")
            if not text:
                return ModelOutput(text="", elapsed_s=time.time() - t0,
                                   error=f"no message content in payload: {str(payload)[:200]}")
        except Exception as e:
            return ModelOutput(text="", elapsed_s=time.time() - t0, error=f"parse: {e}")
        return ModelOutput(text=text, elapsed_s=time.time() - t0)


def make_adapter(spec: str) -> ModelAdapter:
    """Parse adapter spec strings:
       ollama:<model>
       lmstudio:<model>            (OpenAI-compatible /v1/chat/completions)
       lmstudio-api:<model>        (LMStudio /api/v1/chat 'responses' endpoint)
       openrouter:<model>
    """
    if spec.startswith("ollama:"):
        return OllamaAdapter(spec[len("ollama:"):])
    if spec.startswith("lmstudio-api:"):
        return LMStudioRespAPIAdapter(spec[len("lmstudio-api:"):])
    if spec.startswith("lmstudio:"):
        return OpenAICompatAdapter(spec[len("lmstudio:"):], host="http://localhost:1234/v1",
                                   label=f"lmstudio:{spec[len('lmstudio:'):]}")
    if spec.startswith("openrouter:"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY not set")
        return OpenAICompatAdapter(spec[len("openrouter:"):], host="https://openrouter.ai/api/v1",
                                   api_key=api_key, label=f"openrouter:{spec[len('openrouter:'):]}")
    raise SystemExit(f"unknown adapter spec: {spec}")


# ----------------------------------------------------------------------- prompts
def build_user_prompt(entry: dict) -> str:
    parts = []
    if entry.get("in_context_schema"):
        parts.append("Existing schema in scope:\n```taxi\n" + entry["in_context_schema"].strip() + "\n```")
    parts.append("Task: " + entry["prompt"])
    parts.append("Return only the Taxi code that satisfies the task. Do not repeat the existing schema.")
    return "\n\n".join(parts)


def build_stuffed_system(corpus: list[dict], n_examples: int = 6) -> str:
    """Strong context-stuffing recipe: a Taxi primer + N representative valid examples."""
    primer = (
        "Taxi is a strongly-typed schema language used by Orbital (orbitalhq.com).\n"
        "Constructs:\n"
        "  - `type X inherits Y`           — primitive-derived alias (Y ∈ String, Int, Decimal, Boolean, Instant, Date)\n"
        "  - `model X { field : Type }`    — record-shaped data type, fields separated by newlines\n"
        "  - `enum X { A, B, C }`          — closed set of constants\n"
        "  - `service X { operation foo(arg : T) : R }` — RPC-style service\n"
        "  - `annotation X` or `annotation X { field : T }` — metadata applied via `@X(...)`\n"
        "  - `find { Foo[] }`              — TaxiQL query returning an array\n"
        "  - `find { Foo } as { name : T }` — projection rewrite\n"
        "  - `T?` denotes nullable; `T[]` denotes array-of-T.\n"
        "Style:\n"
        "  - Prefer named alias types (`type CustomerId inherits String`) over raw primitives.\n"
        "  - Annotations use `@Name` or `@Name(field = value)` form.\n"
        "  - Closed models use `closed model X { ... }`.\n"
    )
    # pick examples diverse by primary construct
    by_construct: dict[str, list[dict]] = {}
    for r in corpus:
        by_construct.setdefault(r.get("construct", "?"), []).append(r)
    chosen: list[str] = []
    rng = random.Random(7)
    for c in ["model", "type", "service", "enum", "annotation", "query"]:
        pool = by_construct.get(c, [])
        if not pool:
            continue
        # short, well-shaped exemplar
        candidates = [r for r in pool if 60 < len(r["taxi"]) < 350]
        if not candidates:
            candidates = pool
        chosen.append(rng.choice(candidates)["taxi"].strip())
        if len(chosen) >= n_examples:
            break
    examples_block = "\n\n".join(f"Example {i+1}:\n```taxi\n{t}\n```" for i, t in enumerate(chosen))
    return f"{SYSTEM_PROMPT}\n\n{primer}\n{examples_block}"


# ----------------------------------------------------------------------- metrics
_FIELD_PATTERN = re.compile(r"^\s*(?:@\w+(?:\([^)]*\))?\s+)?([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w.]*(?:\?|\[\])?)", re.MULTILINE)
_TOPLEVEL_PATTERN = re.compile(r"\b(model|service|enum|annotation|type)\s+([A-Za-z_]\w*)")


def _extract_fields(taxi: str) -> set[tuple[str, str]]:
    return {(m.group(1), m.group(2).rstrip("?[]")) for m in _FIELD_PATTERN.finditer(taxi)}


def _extract_decls(taxi: str) -> set[tuple[str, str]]:
    return {(m.group(1), m.group(2)) for m in _TOPLEVEL_PATTERN.finditer(taxi)}


@dataclass
class EntryResult:
    id: str
    difficulty: str
    domain: str
    output_text: str
    elapsed_s: float
    error: str | None
    is_valid: bool
    error_count: int
    first_error: str | None
    field_recall: float
    decl_recall: float
    decl_extra: int          # decls in output not in gold (over-spec)
    decl_missing: list[str]  # decl names that should have been there


def score_entry(entry: dict, output_text: str, validator: TaxiValidator, elapsed: float, error: str | None) -> EntryResult:
    cleaned = _strip_fences(output_text)
    in_ctx = entry.get("in_context_schema")
    is_valid, error_count, first_error = False, 0, None
    if cleaned:
        try:
            if in_ctx:
                res: ValidationResult = validator.validate_multi(
                    [("schema.taxi", in_ctx), ("output.taxi", cleaned)]
                )
            else:
                res = validator.validate(cleaned, source_name=f"{entry['id']}.taxi")
            is_valid = res.is_valid
            error_count = res.error_count
            if res.errors:
                first_error = res.errors[0].detailMessage
        except Exception as e:
            first_error = f"validator threw: {e}"
    # gold-vs-output structural metrics
    gold_fields = _extract_fields(entry["gold_taxi"])
    out_fields = _extract_fields(cleaned)
    field_recall = (len(gold_fields & out_fields) / len(gold_fields)) if gold_fields else 1.0
    gold_decls = _extract_decls(entry["gold_taxi"])
    out_decls = _extract_decls(cleaned)
    decl_recall = (len(gold_decls & out_decls) / len(gold_decls)) if gold_decls else 1.0
    decl_extra = max(0, len(out_decls) - len(gold_decls & out_decls))
    decl_missing = [f"{k} {v}" for k, v in (gold_decls - out_decls)]
    return EntryResult(
        id=entry["id"], difficulty=entry["difficulty"], domain=entry["domain"],
        output_text=cleaned, elapsed_s=elapsed, error=error,
        is_valid=is_valid, error_count=error_count, first_error=first_error,
        field_recall=field_recall, decl_recall=decl_recall, decl_extra=decl_extra,
        decl_missing=decl_missing,
    )


# ----------------------------------------------------------------------- runner
def run_eval(adapter: ModelAdapter, benchmark: list[dict], system_prompt: str, validator: TaxiValidator,
             concurrency: int = 4) -> list[EntryResult]:
    results: list[EntryResult] = [None] * len(benchmark)  # type: ignore

    def _one(idx_entry):
        idx, entry = idx_entry
        user = build_user_prompt(entry)
        out = adapter.generate(system_prompt, user)
        return idx, score_entry(entry, out.text, validator, out.elapsed_s, out.error)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, (i, e)) for i, e in enumerate(benchmark)]
        for fut in as_completed(futures):
            i, r = fut.result()
            results[i] = r
            tag = "✓" if r.is_valid else "✗"
            errfrag = f" | err: {r.first_error[:50]}" if r.first_error else ""
            print(f"  {tag} {r.id:10s} ({r.elapsed_s:5.1f}s)  recall={r.decl_recall:.2f}/{r.field_recall:.2f}{errfrag}")
    return results


def summarise(results: list[EntryResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    pass_rate = sum(1 for r in results if r.is_valid) / n
    avg_field_recall = sum(r.field_recall for r in results) / n
    avg_decl_recall = sum(r.decl_recall for r in results) / n
    avg_elapsed = sum(r.elapsed_s for r in results) / n
    by_diff: dict[str, list[EntryResult]] = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r)
    per_diff = {}
    for d, rs in by_diff.items():
        per_diff[d] = {
            "n": len(rs),
            "pass_rate": sum(1 for r in rs if r.is_valid) / len(rs),
            "field_recall": sum(r.field_recall for r in rs) / len(rs),
            "decl_recall": sum(r.decl_recall for r in rs) / len(rs),
        }
    return {
        "n": n,
        "pass_rate": pass_rate,
        "field_recall": avg_field_recall,
        "decl_recall": avg_decl_recall,
        "avg_elapsed_s": avg_elapsed,
        "per_difficulty": per_diff,
    }


def write_output(model_label: str, results: list[EntryResult], summary: dict) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]", "_", model_label)
    p = OUT_DIR / f"eval_{safe}.jsonl"
    with p.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "id": r.id, "difficulty": r.difficulty, "domain": r.domain,
                "is_valid": r.is_valid, "error_count": r.error_count,
                "first_error": r.first_error,
                "field_recall": r.field_recall, "decl_recall": r.decl_recall,
                "decl_extra": r.decl_extra, "decl_missing": r.decl_missing,
                "elapsed_s": r.elapsed_s,
                "output_text": r.output_text, "model_error": r.error,
            }) + "\n")
    summary_p = OUT_DIR / f"eval_summary_{safe}.json"
    summary_p.write_text(json.dumps(summary, indent=2))
    return p


def render_summary_md(per_model: list[tuple[str, dict]]) -> str:
    out = ["# Taxi Benchmark — baseline summary", ""]
    out.append("| Model | n | pass | decl | field | s/q | easy / sa / oe (pass) |")
    out.append("|---|--:|--:|--:|--:|--:|---|")
    for label, s in per_model:
        per = s.get("per_difficulty", {})
        easy = per.get("easy", {}).get("pass_rate", 0) * 100
        sa = per.get("schema_aware", {}).get("pass_rate", 0) * 100
        oe = per.get("open_ended", {}).get("pass_rate", 0) * 100
        out.append(
            f"| {label} | {s['n']} | {s['pass_rate']*100:.0f}% | {s['decl_recall']*100:.0f}% | "
            f"{s['field_recall']*100:.0f}% | {s['avg_elapsed_s']:.1f} | "
            f"{easy:.0f} / {sa:.0f} / {oe:.0f} |"
        )
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------- entrypoint
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=False, default="ollama:qwen2.5-coder:7b,ollama:qwen2.5-coder:32b,lmstudio:qwen3.6-27b-mlx",
                   help="comma-separated adapter specs")
    p.add_argument("--stuff", action="store_true", help="use context-stuffed system prompt for all models")
    p.add_argument("--limit", type=int, default=None, help="limit benchmark entries (for smoke testing)")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--list", action="store_true", help="probe local model servers and exit")
    args = p.parse_args()

    if args.list:
        probe_local_models()
        return

    benchmark = [json.loads(l) for l in BENCH_PATH.read_text().splitlines() if l.strip()]
    if args.limit:
        benchmark = benchmark[: args.limit]
    print(f"benchmark: {len(benchmark)} entries")

    corpus = [json.loads(l) for l in CORPUS_PATH.read_text().splitlines() if l.strip()]
    system_prompt = build_stuffed_system(corpus) if args.stuff else SYSTEM_PROMPT
    if args.stuff:
        print(f"using context-stuffed system prompt ({len(system_prompt)} chars, ~{len(system_prompt)//4} tokens)")

    summaries: list[tuple[str, dict]] = []
    with TaxiValidator() as validator:
        for spec in args.models.split(","):
            adapter = make_adapter(spec.strip())
            label = adapter.name + (".stuffed" if args.stuff else ".plain")
            print(f"\n=== {label} ===")
            t0 = time.time()
            results = run_eval(adapter, benchmark, system_prompt, validator, concurrency=args.concurrency)
            summary = summarise(results)
            print(f"  total elapsed: {time.time()-t0:.1f}s")
            print(f"  pass_rate: {summary['pass_rate']*100:.1f}%   "
                  f"decl_recall: {summary['decl_recall']*100:.1f}%   "
                  f"field_recall: {summary['field_recall']*100:.1f}%")
            for d, ds in summary["per_difficulty"].items():
                print(f"    {d:14s}  pass={ds['pass_rate']*100:.0f}%  decl={ds['decl_recall']*100:.0f}%  field={ds['field_recall']*100:.0f}%")
            write_output(label, results, summary)
            summaries.append((label, summary))

    md = render_summary_md(summaries)
    summary_path = OUT_DIR / "eval_summary.md"
    summary_path.write_text(md)
    print(f"\n=== summary: {summary_path.relative_to(REPO)} ===")
    print(md)


def probe_local_models():
    """List which adapters look reachable right now."""
    print("=== adapter probe ===")
    try:
        with _urlreq.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            tags = json.loads(r.read()).get("models", [])
            print(f"ollama: {len(tags)} models")
            for t in tags: print(f"  - {t['name']}")
    except Exception as e:
        print(f"ollama: unreachable ({e})")
    try:
        with _urlreq.urlopen("http://localhost:1234/v1/models", timeout=2) as r:
            data = json.loads(r.read()).get("data", [])
            print(f"lmstudio: {len(data)} models")
            for d in data: print(f"  - {d['id']}")
    except Exception as e:
        print(f"lmstudio: unreachable ({e})")


if __name__ == "__main__":
    main()
