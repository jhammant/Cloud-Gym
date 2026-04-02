"""Run zero-shot LLM repair baseline evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
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


def make_repair_fn(model_name: str):
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
        # Strip markdown fences
        lines = response.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    return repair


async def run_baselines(
    benchmark_path: str,
    output_dir: str,
    models: list[str],
    n_attempts: int,
):
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(benchmark_path)
    console.print(f"[bold]Benchmark: {len(evaluator.dataset)} entries[/bold]")

    for model in models:
        console.print(f"\n[bold]Evaluating {model}...[/bold]")
        repair_fn = make_repair_fn(model)

        report = await evaluator.evaluate_model(
            model_fn=repair_fn,
            model_name=model,
            n_attempts=n_attempts,
            k_values=[1, 3],
        )

        # Display results
        table = Table(title=f"Results: {model}")
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

        # Save report
        report_path = results_dir / f"{model.replace(':', '_')}.json"
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


@click.command()
@click.option(
    "--benchmark",
    default=str(BENCHMARK_DIR / "benchmark.jsonl"),
    help="Path to benchmark.jsonl",
)
@click.option(
    "--output-dir",
    default=str(BENCHMARK_DIR / "results"),
    help="Directory for result JSON files",
)
@click.option(
    "--models",
    default="deepseek-r1:1.5b,qwen2.5-coder:7b",
    help="Comma-separated list of Ollama models to evaluate",
)
@click.option("--n-attempts", default=5, help="Number of repair attempts per entry")
def main(benchmark: str, output_dir: str, models: str, n_attempts: int):
    """Run baseline evaluations on the benchmark."""
    model_list = [m.strip() for m in models.split(",")]
    asyncio.run(run_baselines(benchmark, output_dir, model_list, n_attempts))


if __name__ == "__main__":
    main()
