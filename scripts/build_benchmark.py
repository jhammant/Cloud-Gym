"""Build curated benchmark subset from test split."""

from __future__ import annotations

import logging

import click
from rich.console import Console

from cloudgym.benchmark.dataset import BenchmarkDataset
from cloudgym.utils.config import BENCHMARK_DIR, TRAINING_DIR

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@click.command()
@click.option(
    "--test-jsonl",
    default=str(TRAINING_DIR / "test.jsonl"),
    help="Path to test.jsonl from training data generation",
)
@click.option(
    "--output",
    default=str(BENCHMARK_DIR / "benchmark.jsonl"),
    help="Path to write benchmark.jsonl",
)
@click.option("--target-size", default=200, help="Target number of benchmark entries")
def main(test_jsonl: str, output: str, target_size: int):
    """Build curated benchmark dataset from test split."""
    console.print(f"[bold]Building benchmark...[/bold]")
    console.print(f"  Source: {test_jsonl}")
    console.print(f"  Output: {output}")
    console.print(f"  Target size: {target_size}")

    dataset = BenchmarkDataset.build(
        test_jsonl=test_jsonl,
        output_path=output,
        target_size=target_size,
    )

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Benchmark entries: {len(dataset)}")


if __name__ == "__main__":
    main()
