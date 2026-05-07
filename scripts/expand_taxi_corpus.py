"""P2: expand the valid Taxi corpus by walking upstream .taxi files.

Sources:
  - data/upstream/taxilang/**/*.taxi  (test fixtures, sample compiled taxonomies)
  - data/upstream/orbital/**/*.taxi   (test fixtures, regression taxonomies)

Each candidate is validated in isolation via the strict validator. Files that
reference types from sibling files in their original package will fail strict
validation here — that's the right behaviour. We're after self-contained
snippets suitable for training units, not full multi-file packages.

Output:
  data/taxi/valid_corpus.jsonl   merged with the existing 299 inline-mined
                                 snippets, deduped by content hash, stratified
                                 by primary construct.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from cloudgym.taxi.validator import TaxiValidator

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "data/upstream"
OUT_PATH = REPO / "data/taxi/valid_corpus.jsonl"

# Skip files that are unlikely to produce useful training units.
MIN_LINES = 2
MAX_LINES = 500   # very large files are aggregations; harder to use as units
MAX_BYTES = 50_000

# Construct labels (primary construct in the snippet)
CONSTRUCT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("query", re.compile(r"\b(query|find\s*\{)\b")),
    ("projection", re.compile(r"\bproject(ion)?\b|\bas\s*\{")),
    ("service", re.compile(r"\bservice\s+\w+")),
    ("annotation", re.compile(r"\bannotation\s+\w+")),
    ("enum", re.compile(r"\benum\s+\w+")),
    ("model", re.compile(r"\bmodel\s+\w+")),
    ("type", re.compile(r"\btype\s+\w+")),
]


def _classify(taxi: str) -> str:
    for label, pat in CONSTRUCT_PATTERNS:
        if pat.search(taxi):
            return label
    return "other"


def _content_hash(taxi: str) -> str:
    # normalise trailing whitespace + line endings before hashing
    norm = "\n".join(line.rstrip() for line in taxi.replace("\r\n", "\n").splitlines())
    return hashlib.sha1(norm.encode()).hexdigest()


def _walk_taxi_files() -> Iterable[Path]:
    for path in UPSTREAM.rglob("*.taxi"):
        if not path.is_file():
            continue
        # skip the upstream test-duplicate fixture used by the compiler tests
        if path.name == "test-duplicate.taxi":
            continue
        yield path


def _load_existing_corpus() -> list[dict]:
    if not OUT_PATH.exists():
        return []
    return [json.loads(l) for l in OUT_PATH.read_text().splitlines() if l.strip()]


def main() -> None:
    existing = _load_existing_corpus()
    print(f"existing corpus: {len(existing)} snippets")

    # build hash → existing record map (don't drop inline-sourced records)
    by_hash: dict[str, dict] = {}
    for r in existing:
        by_hash[_content_hash(r["taxi"])] = r

    print(f"deduped existing: {len(by_hash)} unique")

    # walk upstream .taxi files
    candidates: list[tuple[Path, str]] = []
    for path in _walk_taxi_files():
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        if len(text.encode()) > MAX_BYTES:
            continue
        line_count = text.count("\n") + 1
        if line_count < MIN_LINES or line_count > MAX_LINES:
            continue
        if not text.strip():
            continue
        candidates.append((path, text))
    print(f"candidate .taxi files: {len(candidates)}")

    # validate each — drop any with strict-mode errors
    file_added = 0
    file_failed_validate = 0
    file_dup = 0
    by_source = Counter()
    failed_paths: list[Path] = []
    with TaxiValidator() as v:
        for path, text in candidates:
            h = _content_hash(text)
            if h in by_hash:
                file_dup += 1
                continue
            try:
                res = v.validate(text, source_name=path.name)
            except Exception as e:
                file_failed_validate += 1
                failed_paths.append(path)
                continue
            if not res.is_valid:
                file_failed_validate += 1
                failed_paths.append(path)
                continue
            rel = path.relative_to(REPO)
            by_hash[h] = {
                "taxi": text.rstrip() + "\n",
                "source_file": str(rel),
                "source_line": 1,
                "via_var": None,
                "origin": "file",
            }
            by_source[str(rel.parts[2])] += 1   # taxilang | orbital
            file_added += 1

        # ---- second pass: try compiling sibling files together as packages ----
        # Many .taxi files reference types from siblings in the same directory.
        # Concatenate per-directory groups (size 2..8) and accept the combined
        # source as one snippet if it validates. This recovers schema-set
        # examples that don't compile in isolation.
        from collections import defaultdict
        groups: dict[Path, list[Path]] = defaultdict(list)
        for p in failed_paths:
            groups[p.parent].append(p)
        package_added = 0
        package_total_files = 0
        package_failed = 0
        for parent, files in groups.items():
            if not 2 <= len(files) <= 12:
                continue
            files_sorted = sorted(files)
            sources: list[tuple[str, str]] = []
            for fp in files_sorted:
                try:
                    content = fp.read_text(errors="replace")
                    if content.strip():
                        sources.append((fp.name, content))
                except Exception:
                    pass
            if not sources:
                continue
            total_bytes = sum(len(c.encode()) for _, c in sources)
            if total_bytes > MAX_BYTES * 4:
                continue
            try:
                res = v.validate_multi(sources)
            except Exception:
                package_failed += 1; continue
            if not res.is_valid:
                package_failed += 1; continue
            # store as a labelled multi-source unit
            combined = "\n// ---\n".join(c.rstrip() for _, c in sources)
            h = _content_hash(combined)
            if h in by_hash:
                continue
            rel_parent = parent.relative_to(REPO)
            by_hash[h] = {
                "taxi": combined + "\n",
                "source_file": str(rel_parent) + "/[package]",
                "source_line": 1,
                "via_var": None,
                "origin": "package",
                "package_files": [n for n, _ in sources],
            }
            by_source[str(rel_parent.parts[2])] += 1
            package_added += 1
            package_total_files += len(sources)
        print(f"  package pass: groups={len(groups)} added={package_added} files_in_added={package_total_files} failed={package_failed}")

    # stratification labels
    final = list(by_hash.values())
    for r in final:
        r["construct"] = _classify(r["taxi"])
        r.setdefault("origin", "inline")

    # write back
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    print()
    print("=== expansion report ===")
    print(f"  files scanned:              {len(candidates)}")
    print(f"  added from files:           {file_added}")
    print(f"  duplicates of inline:       {file_dup}")
    print(f"  failed strict validation:   {file_failed_validate}")
    print(f"  by source repo: {dict(by_source)}")
    print()
    print(f"=== final corpus: {len(final)} snippets -> {OUT_PATH.relative_to(REPO)} ===")
    print()
    print("by origin:")
    for k, v in Counter(r["origin"] for r in final).most_common():
        print(f"  {v:4d}  {k}")
    print()
    print("by primary construct:")
    for k, v in Counter(r["construct"] for r in final).most_common():
        print(f"  {v:4d}  {k}")
    avg_chars = sum(len(r["taxi"]) for r in final) / max(1, len(final))
    print()
    print(f"avg chars: {avg_chars:.0f}")
    sizes = sorted(len(r["taxi"]) for r in final)
    if sizes:
        print(f"size p50: {sizes[len(sizes)//2]}, p90: {sizes[int(len(sizes)*0.9)]}, max: {sizes[-1]}")


if __name__ == "__main__":
    main()
