"""Convert Cloud-Gym training data to MLX chat fine-tuning format.

Reads JSONL records from the pipeline and outputs chat-formatted JSONL
suitable for mlx_lm.lora fine-tuning.

Output format (one per line):
{"messages": [
  {"role": "system", "content": "Fix the broken Terraform config..."},
  {"role": "user",   "content": "Errors:\\n...\\n\\nBroken config:\\n..."},
  {"role": "assistant", "content": "<corrected config>"}
]}
"""

from __future__ import annotations

import json
from pathlib import Path

import click


SYSTEM_PROMPT = (
    "You are an Infrastructure-as-Code repair assistant. "
    "Fix the broken Terraform configuration below. "
    "Return ONLY the corrected HCL configuration with no explanation."
)


def _format_record(record: dict) -> dict:
    """Convert a single pipeline record to MLX chat format."""
    errors = record.get("errors", [])
    warnings = record.get("warnings", [])
    broken = record["broken_config"]
    gold = record["gold_config"]

    # Build user message with error context
    parts = []
    if errors:
        parts.append("Errors:\n" + "\n".join(f"- {e}" for e in errors))
    if warnings:
        parts.append("Warnings:\n" + "\n".join(f"- {w}" for w in warnings))
    parts.append(f"Broken config:\n```hcl\n{broken}\n```")

    user_content = "\n\n".join(parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gold},
        ]
    }


@click.command()
@click.option(
    "--input-dir", "-i",
    default="data/training",
    help="Directory with train.jsonl, val.jsonl, test.jsonl from pipeline",
)
@click.option(
    "--output-dir", "-o",
    default="data/finetune",
    help="Output directory for MLX chat-formatted JSONL",
)
def main(input_dir: str, output_dir: str):
    """Convert pipeline training data to MLX fine-tuning format."""
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Map: pipeline uses "val", MLX expects "valid"
    split_map = {"train": "train", "valid": "val", "test": "test"}
    for out_name, src_name in split_map.items():
        src = inp / f"{src_name}.jsonl"
        if not src.exists():
            click.echo(f"Skipping {out_name} (not found)")
            continue

        records = [json.loads(line) for line in src.read_text().strip().split("\n") if line.strip()]
        formatted = [_format_record(r) for r in records]

        dst = out / f"{out_name}.jsonl"
        with open(dst, "w") as f:
            for item in formatted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        click.echo(f"{out_name}: {len(formatted)} records -> {dst}")

    click.echo("Done!")


if __name__ == "__main__":
    main()
