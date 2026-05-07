"""Combine all eval_summary_*.json files into one cross-model report."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/taxi"


def main() -> None:
    rows: list[tuple[str, dict]] = []
    for p in sorted(OUT.glob("eval_summary_*.json")):
        # filename pattern: eval_summary_<safe_label>.json
        label = p.stem.replace("eval_summary_", "")
        # rehydrate label-ish string: "ollama_qwen2_5_coder_7b_plain" -> readable form
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        rows.append((label, data))
    if not rows:
        raise SystemExit("no eval_summary_*.json files found — run scripts/eval_taxi.py first")

    # split by suffix .plain / .stuffed for side-by-side
    def _readable(label: str) -> str:
        # e.g. ollama_qwen2_5_coder_7b_plain → ollama:qwen2.5-coder:7b plain
        s = label
        s = s.replace("ollama_", "ollama:")
        s = s.replace("lmstudio_api_", "lmstudio-api:")
        s = s.replace("lmstudio_", "lmstudio:")
        s = s.replace("openrouter_", "openrouter:")
        s = re.sub(r"_(plain|stuffed)$", r" [\1]", s)
        s = re.sub(r"qwen2_5_coder_(\d+b)", r"qwen2.5-coder:\1", s)
        s = s.replace("qwen_qwen3_coder_next", "qwen/qwen3-coder-next")
        s = s.replace("qwen3_6_27b_mlx", "qwen3.6-27b-mlx")
        return s

    out: list[str] = ["# Taxi Benchmark — combined report", ""]
    out.append("All models scored against `data/taxi/benchmark.jsonl` (100 entries:")
    out.append("40 easy, 30 schema_aware, 30 open_ended across 11 domains).")
    out.append("")
    out.append("Metrics:")
    out.append("- **pass** = compile pass rate (strict taxilang validator)")
    out.append("- **decl** = top-level declaration recall vs gold")
    out.append("- **field** = field name+type recall vs gold")
    out.append("- **easy / sa / oe** = pass rate per difficulty bucket")
    out.append("")
    out.append("| Model | n | pass | decl | field | s/q | easy / sa / oe |")
    out.append("|---|--:|--:|--:|--:|--:|---|")
    # sort: plain models first, then stuffed
    rows.sort(key=lambda x: (x[0].endswith("_stuffed"), x[0]))
    for label, s in rows:
        per = s.get("per_difficulty", {})
        easy = per.get("easy", {}).get("pass_rate", 0) * 100
        sa = per.get("schema_aware", {}).get("pass_rate", 0) * 100
        oe = per.get("open_ended", {}).get("pass_rate", 0) * 100
        out.append(
            f"| {_readable(label)} | {s['n']} | {s['pass_rate']*100:.0f}% | "
            f"{s['decl_recall']*100:.0f}% | {s['field_recall']*100:.0f}% | "
            f"{s['avg_elapsed_s']:.1f} | {easy:.0f} / {sa:.0f} / {oe:.0f} |"
        )

    # delta if both plain + stuffed present for the same model
    plain = {r[0].rsplit("_plain", 1)[0]: r[1] for r in rows if r[0].endswith("_plain")}
    stuffed = {r[0].rsplit("_stuffed", 1)[0]: r[1] for r in rows if r[0].endswith("_stuffed")}
    deltas = []
    for k in sorted(plain.keys() & stuffed.keys()):
        p, s = plain[k], stuffed[k]
        deltas.append((k, p["pass_rate"], s["pass_rate"]))
    if deltas:
        out.append("")
        out.append("## Context-stuffing delta (pass rate)")
        out.append("")
        out.append("| Model | plain | stuffed | Δ |")
        out.append("|---|--:|--:|--:|")
        for k, pp, sp in deltas:
            out.append(f"| {_readable(k)} | {pp*100:.0f}% | {sp*100:.0f}% | +{(sp-pp)*100:.0f}pp |")

    text = "\n".join(out) + "\n"
    (OUT / "eval_combined.md").write_text(text)
    print(text)
    print(f"\nwritten: {(OUT/'eval_combined.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
