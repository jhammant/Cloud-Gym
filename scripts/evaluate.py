"""Full evaluation pipeline — Ollama baselines + MLX fine-tuned models."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

from cloudgym.benchmark.evaluator import Evaluator
from cloudgym.utils.config import BENCHMARK_DIR, InverterConfig
from cloudgym.utils.ollama import OllamaClient

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = (
    "You are an expert Infrastructure-as-Code engineer. "
    "Fix the broken configuration below. Return ONLY the fixed configuration "
    "with no explanation, no markdown fences, and no comments about the fix."
)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences and chat tokens from model output."""
    # Truncate at first stop token
    for stop in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
        if stop in text:
            text = text[:text.index(stop)]

    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def make_ollama_repair_fn(model_name: str):
    """Create a repair function for a given Ollama model."""
    config = InverterConfig(ollama_model=model_name)
    client = OllamaClient(config=config)

    async def repair(broken_config: str, errors: list[str]) -> str:
        error_text = "\n".join(errors) if errors else "Unknown error"
        prompt = (
            f"This IaC configuration has validation errors:\n\n"
            f"Errors:\n{error_text}\n\n"
            f"Broken config:\n```\n{broken_config}\n```\n\n"
            f"Return the fixed configuration:"
        )
        response = await client.generate(prompt, system=REPAIR_SYSTEM_PROMPT)
        return _strip_markdown_fences(response)

    return repair


def make_lmstudio_repair_fn(model_name: str, base_url: str = "http://localhost:1234"):
    """Create a repair function for a model served via LM Studio."""

    async def repair(broken_config: str, errors: list[str]) -> str:
        error_text = "\n".join(errors) if errors else "Unknown error"
        prompt = (
            f"This IaC configuration has validation errors:\n\n"
            f"Errors:\n{error_text}\n\n"
            f"Broken config:\n```\n{broken_config}\n```\n\n"
            f"Return the fixed configuration:"
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url}/api/v1/chat",
                json={
                    "model": model_name,
                    "system_prompt": REPAIR_SYSTEM_PROMPT,
                    "input": prompt,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # LM Studio returns output as list of {type, content}
            parts = data.get("output", [])
            message = ""
            for part in parts:
                if part.get("type") == "message":
                    message = part.get("content", "")
                    break
            return _strip_markdown_fences(message)

    return repair


def make_mlx_repair_fn(model_path: str, adapter_path: str):
    """Create a repair function using an MLX fine-tuned model."""

    async def repair(broken_config: str, errors: list[str]) -> str:
        # Lazy-load model and tokenizer on first call
        if not hasattr(repair, "_model"):
            from mlx_lm import load

            repair._model, repair._tokenizer = load(
                model_path, adapter_path=adapter_path
            )

        from mlx_lm import generate

        error_text = "\n".join(errors) if errors else "Unknown error"
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"This IaC configuration has validation errors:\n\n"
                    f"Errors:\n{error_text}\n\n"
                    f"Broken config:\n```\n{broken_config}\n```\n\n"
                    f"Return the fixed configuration:"
                ),
            },
        ]
        prompt_text = repair._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.3)
        response = generate(
            repair._model, repair._tokenizer, prompt=prompt_text,
            max_tokens=2048, sampler=sampler,
        )
        return _strip_markdown_fences(response)

    return repair


def _display_report(report) -> None:
    """Display a single model's evaluation report as a Rich table."""
    table = Table(title=f"Results: {report.model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for k, v in report.pass_at_k.items():
        table.add_row(f"pass@{k}", f"{v:.3f}")

    table.add_section()
    for cat, metrics in sorted(report.per_category.items()):
        for k, v in metrics.items():
            table.add_row(f"{cat} pass@{k}", f"{v:.3f}")

    table.add_section()
    for diff, metrics in sorted(report.per_difficulty.items()):
        for k, v in metrics.items():
            table.add_row(f"{diff} pass@{k}", f"{v:.3f}")

    console.print(table)


def _save_report(report, results_dir: Path, name: str) -> None:
    """Save an evaluation report as JSON."""
    report_path = results_dir / f"{name.replace(':', '_').replace('/', '_')}.json"
    report_data = {
        "model_name": report.model_name,
        "n_attempts": report.n_attempts,
        "total_entries": report.total_entries,
        "pass_at_k": {str(k): v for k, v in report.pass_at_k.items()},
        "per_category": {
            cat: {str(k): v for k, v in metrics.items()}
            for cat, metrics in report.per_category.items()
        },
        "per_difficulty": {
            diff: {str(k): v for k, v in metrics.items()}
            for diff, metrics in report.per_difficulty.items()
        },
        "per_format": {
            fmt: {str(k): v for k, v in metrics.items()}
            for fmt, metrics in report.per_format.items()
        },
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    console.print(f"  Saved to {report_path}")


async def run_evaluation(
    benchmark_path: str,
    output_dir: str,
    models: list[str],
    adapter_path: str,
    base_model: str,
    n_attempts: int,
    skip_baselines: bool = False,
    skip_finetuned: bool = False,
    lmstudio_models: list[str] | None = None,
    lmstudio_url: str = "http://localhost:1234",
):
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(benchmark_path)
    console.print(f"[bold]Benchmark: {len(evaluator.dataset)} entries[/bold]")

    all_reports = []

    # Ollama baselines
    if not skip_baselines:
        for model in models:
            console.print(f"\n[bold]Evaluating {model}...[/bold]")
            report = await evaluator.evaluate_model(
                model_fn=make_ollama_repair_fn(model),
                model_name=model,
                n_attempts=n_attempts,
                k_values=[1, 3],
            )
            _display_report(report)
            _save_report(report, results_dir, model)
            all_reports.append(report)

    # MLX fine-tuned model
    if not skip_finetuned and Path(adapter_path).exists():
        console.print(f"\n[bold]Evaluating fine-tuned ({adapter_path})...[/bold]")
        report = await evaluator.evaluate_model(
            model_fn=make_mlx_repair_fn(base_model, adapter_path),
            model_name=f"finetuned:{Path(adapter_path).name}",
            n_attempts=n_attempts,
            k_values=[1, 3],
        )
        _display_report(report)
        _save_report(report, results_dir, report.model_name)
        all_reports.append(report)
    elif not skip_finetuned:
        console.print(f"[yellow]Adapter not found at {adapter_path}, skipping.[/yellow]")

    # LM Studio models
    if lmstudio_models:
        for model in lmstudio_models:
            console.print(f"\n[bold]Evaluating {model} (LM Studio)...[/bold]")
            report = await evaluator.evaluate_model(
                model_fn=make_lmstudio_repair_fn(model, lmstudio_url),
                model_name=model,
                n_attempts=n_attempts,
                k_values=[1, 3],
            )
            _display_report(report)
            _save_report(report, results_dir, model)
            all_reports.append(report)

    # Comparison table
    if len(all_reports) > 1:
        comp = Table(title="Model Comparison")
        comp.add_column("Model", style="cyan")
        comp.add_column("pass@1", justify="right", style="green")
        comp.add_column("pass@3", justify="right", style="green")
        for r in all_reports:
            comp.add_row(
                r.model_name,
                f"{r.pass_at_k.get(1, 0):.3f}",
                f"{r.pass_at_k.get(3, 0):.3f}",
            )
        console.print(comp)


@click.command()
@click.option("--benchmark", default=str(BENCHMARK_DIR / "benchmark.jsonl"), help="Path to benchmark.jsonl")
@click.option("--output-dir", default=str(BENCHMARK_DIR / "results"), help="Directory for result JSON files")
@click.option("--models", default="llama3.2:3b,qwen2.5-coder:7b,qwen2.5-coder:32b", help="Comma-separated Ollama models")
@click.option("--adapter-path", default="data/models/iac-repair-adapter", help="Path to MLX adapter weights")
@click.option("--base-model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit", help="MLX base model ID")
@click.option("--n-attempts", default=5, help="Number of repair attempts per entry")
@click.option("--skip-baselines", is_flag=True, help="Skip Ollama baseline evaluation")
@click.option("--skip-finetuned", is_flag=True, help="Skip fine-tuned model evaluation")
@click.option("--lmstudio-models", default=None, help="Comma-separated LM Studio model IDs")
@click.option("--lmstudio-url", default="http://localhost:1234", help="LM Studio API base URL")
def main(benchmark, output_dir, models, adapter_path, base_model, n_attempts, skip_baselines, skip_finetuned, lmstudio_models, lmstudio_url):
    """Run full evaluation pipeline — baselines, fine-tuned, and LM Studio models."""
    model_list = [m.strip() for m in models.split(",")]
    lmstudio_list = [m.strip() for m in lmstudio_models.split(",")] if lmstudio_models else None
    asyncio.run(run_evaluation(
        benchmark, output_dir, model_list, adapter_path, base_model,
        n_attempts, skip_baselines, skip_finetuned, lmstudio_list, lmstudio_url,
    ))


if __name__ == "__main__":
    main()
