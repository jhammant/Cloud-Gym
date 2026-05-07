"""Mine valid (schema + query) compile units from taxilang test suite.

Targets the construct deficit identified in P2: only 20 query and 2 projection
examples in the inline corpus. These come from Kotlin tests of the form:

    "<schema>".compiledWithQuery("<query>")

which compile schema + query together and expect success. We capture both
sources as a single multi-source unit and add to valid_corpus.jsonl.

Patterns captured (chained off a triple-quoted Kotlin string with a triple-quoted
Taxi argument):
  - .compiledWithQuery
  - .compiledWithMutation
  - .compiledWithStreamingQuery
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from cloudgym.taxi.validator import TaxiValidator

REPO = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO / "data/upstream/taxilang/compiler/src/test"
OUT_PATH = REPO / "data/taxi/valid_corpus.jsonl"

TRIPLE = '"""'
SUCCESS_QUERY_CALLS = {
    "compiledWithQuery",
    "compiledWithMutation",
    "compiledWithStreamingQuery",
}


def _find_triples(text: str) -> list[tuple[int, int, str]]:
    out = []
    i = 0
    while True:
        j = text.find(TRIPLE, i)
        if j == -1: break
        k = text.find(TRIPLE, j + 3)
        if k == -1: break
        out.append((j, k + 3, text[j + 3 : k]))
        i = k + 3
    return out


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def kotlin_trim_indent(s: str) -> str:
    lines = s.split("\n")
    while lines and lines[0].strip() == "": lines.pop(0)
    while lines and lines[-1].strip() == "": lines.pop()
    if not lines: return ""
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    common = min(indents) if indents else 0
    return "\n".join(ln[common:] if len(ln) >= common else ln for ln in lines)


_CHAIN_HEAD = re.compile(
    r"\s*(?:\.\s*trimIndent\(\)\s*)?\.\s*(?P<call>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def _next_chain_after(text: str, pos: int) -> tuple[str, int, int] | None:
    """Match a `.<name>(` call chained off the position. Return (name, args_start, args_end_paren)."""
    m = _CHAIN_HEAD.match(text, pos)
    if not m: return None
    args_start = m.end()
    depth = 1
    i = args_start
    in_str: str | None = None
    while i < len(text) and depth > 0:
        c = text[i]
        if in_str is not None:
            if c == "\\" and i + 1 < len(text):
                i += 2; continue
            if c == in_str:
                in_str = None
        elif c == '"':
            # detect triple-quote
            if text[i:i+3] == TRIPLE:
                # skip the whole triple block
                end = text.find(TRIPLE, i + 3)
                if end == -1: return None
                i = end + 3; continue
            in_str = '"'
        elif c == "'":
            in_str = "'"
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    if depth != 0: return None
    return m.group("call"), args_start, i - 1


def _arg_triple(args: str) -> str | None:
    j = args.find(TRIPLE)
    if j == -1: return None
    k = args.find(TRIPLE, j + 3)
    if k == -1: return None
    return args[j + 3 : k]


def mine_file(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    rel = path.relative_to(REPO)
    triples = _find_triples(text)
    out: list[dict] = []
    for open_idx, after_idx, content in triples:
        first = _next_chain_after(text, after_idx)
        if not first: continue
        call, args_start, args_end = first
        if call not in SUCCESS_QUERY_CALLS: continue
        args_text = text[args_start:args_end]
        inner = _arg_triple(args_text)
        if inner is None: continue
        schema = kotlin_trim_indent(content)
        query = kotlin_trim_indent(inner)
        out.append({
            "schema": schema,
            "query": query,
            "source_file": str(rel),
            "source_line": _line_of(text, open_idx),
            "call": call,
        })
    return out


def main() -> None:
    if not TEST_ROOT.exists():
        raise SystemExit(f"taxilang test root not found: {TEST_ROOT}")

    raw_pairs: list[dict] = []
    for kt in sorted(TEST_ROOT.rglob("*.kt")):
        raw_pairs.extend(mine_file(kt))
    print(f"raw schema+query pairs found: {len(raw_pairs)}")

    # validate each combined source via multi-source endpoint
    existing = []
    if OUT_PATH.exists():
        existing = [json.loads(l) for l in OUT_PATH.read_text().splitlines() if l.strip()]
    by_hash: dict[str, dict] = {}
    for r in existing:
        h = hashlib.sha1(r["taxi"].encode()).hexdigest()
        by_hash[h] = r
    pre_count = len(by_hash)

    added = 0
    failed = 0
    by_call = Counter()
    with TaxiValidator() as v:
        for p in raw_pairs:
            sources = [
                ("schema.taxi", p["schema"]),
                ("query.taxi", p["query"]),
            ]
            try:
                res = v.validate_multi(sources)
            except Exception:
                failed += 1; continue
            if not res.is_valid:
                failed += 1; continue
            combined = f"{p['schema'].rstrip()}\n\n// --- query ---\n{p['query'].rstrip()}\n"
            h = hashlib.sha1(combined.encode()).hexdigest()
            if h in by_hash: continue
            by_hash[h] = {
                "taxi": combined,
                "source_file": p["source_file"],
                "source_line": p["source_line"],
                "via_var": None,
                "origin": f"query/{p['call']}",
                "construct": "query" if "find {" in p["query"] or "given {" in p["query"] else "projection",
            }
            by_call[p["call"]] += 1
            added += 1

    final = list(by_hash.values())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== query-pattern mining ===")
    print(f"  raw candidates:       {len(raw_pairs)}")
    print(f"  added:                {added}")
    print(f"  failed validation:    {failed}")
    print(f"  by call pattern:      {dict(by_call)}")
    print(f"  corpus before/after:  {pre_count} -> {len(final)}")


if __name__ == "__main__":
    main()
