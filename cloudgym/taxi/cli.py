"""taxify CLI: generate Taxi schema code from natural-language descriptions.

Usage:
    taxify "a Customer model with id and email"
    taxify "an Order service" --schema schema.taxi
    cat prompt.txt | taxify -
    taxify "..." --backend mlx --adapter data/models/taxi-nl-adapter-v2
    taxify "..." --backend gguf --model taxi-nl-3b-q4.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from cloudgym.taxi.generator import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_BASE_MODEL,
    GGUFGenerator,
    MLXGenerator,
    resolve_gguf_model,
)

console = Console()
stderr = Console(stderr=True)


def _make_generator(backend: str, model: str | None, adapter: str | None):
    if backend == "mlx":
        return MLXGenerator(
            base_model=DEFAULT_BASE_MODEL,
            adapter_path=adapter or DEFAULT_ADAPTER_PATH,
        )
    if backend == "gguf":
        path = resolve_gguf_model(model)
        return GGUFGenerator(gguf_path=path)
    raise click.UsageError(f"unknown backend: {backend}")


def _read_prompt(prompt_arg: str) -> str:
    if prompt_arg == "-":
        return sys.stdin.read().strip()
    p = Path(prompt_arg)
    if p.exists() and p.is_file():
        return p.read_text().strip()
    return prompt_arg.strip()


def _read_schema(schema_arg: str | None) -> Optional[str]:
    if schema_arg is None:
        return None
    if schema_arg == "-":
        return sys.stdin.read()
    p = Path(schema_arg)
    if p.exists():
        return p.read_text()
    return schema_arg


@click.command()
@click.argument("prompt", required=True)
@click.option("--schema", "-s", default=None,
              help="Existing in-context schema (file path, '-' for stdin, or literal Taxi).")
@click.option("--backend", "-b", default="gguf", type=click.Choice(["mlx", "gguf"]),
              help="Inference backend (default: gguf — runs anywhere on CPU).")
@click.option("--model", "-m", default=None,
              help="GGUF file path (auto-downloads taxi-nl-3b-q4.gguf if omitted).")
@click.option("--adapter", "-a", default=None,
              help="MLX LoRA adapter directory (default: data/models/taxi-nl-adapter-v2).")
@click.option("--validate/--no-validate", default=False,
              help="Run output through the strict taxilang validator and report errors.")
@click.option("--output", "-o", default=None,
              help="Write output to file instead of stdout.")
def main(
    prompt: str,
    schema: str | None,
    backend: str,
    model: str | None,
    adapter: str | None,
    validate: bool,
    output: str | None,
) -> None:
    """Translate a natural-language description into Taxi schema code."""
    prompt_text = _read_prompt(prompt)
    schema_text = _read_schema(schema)

    if not prompt_text:
        raise click.UsageError("empty prompt")

    gen = _make_generator(backend, model, adapter)
    stderr.print(f"[dim]generating ({backend})...[/dim]")
    taxi = gen.generate(prompt_text, schema=schema_text).rstrip() + "\n"

    if validate:
        try:
            from cloudgym.taxi.validator import TaxiValidator
            with TaxiValidator() as v:
                if schema_text:
                    res = v.validate_multi([("schema.taxi", schema_text), ("output.taxi", taxi)])
                else:
                    res = v.validate(taxi)
            tag = "[green]✓ valid[/green]" if res.is_valid else f"[red]✗ {res.error_count} errors[/red]"
            stderr.print(f"[dim]validate: {tag}[/dim]")
            for e in res.errors[:5]:
                if e.severity.lower() == "error":
                    stderr.print(f"  [red]L{e.line}C{e.char}[/red] {e.detailMessage[:120]}")
        except Exception as e:
            stderr.print(f"[yellow]validate skipped: {e}[/yellow]")

    if output:
        Path(output).write_text(taxi)
        stderr.print(f"[dim]wrote {len(taxi)} bytes -> {output}[/dim]")
    else:
        sys.stdout.write(taxi)


if __name__ == "__main__":
    main()
