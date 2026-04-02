"""Generate training data from gold configs."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console

from cloudgym.generator.pipeline import PipelineRunner
from cloudgym.utils.config import GOLD_DIR, TRAINING_DIR, PipelineConfig

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def run_generate(
    gold_dir: str,
    output_dir: str,
    programmatic_variants: int,
    agentic_variants: int,
    skip_agentic: bool,
):
    runner = PipelineRunner(
        config=PipelineConfig(
            programmatic_variants=programmatic_variants,
            agentic_variants=agentic_variants,
        ),
    )

    console.print(f"[bold]Generating training data...[/bold]")
    console.print(f"  Gold dir: {gold_dir}")
    console.print(f"  Output dir: {output_dir}")
    console.print(f"  Programmatic variants: {programmatic_variants}")
    console.print(f"  Agentic variants: {agentic_variants}")
    console.print(f"  Skip agentic: {skip_agentic}")

    metadata = await runner.run(
        gold_dir=gold_dir,
        output_dir=output_dir,
        programmatic_variants=programmatic_variants,
        agentic_variants=agentic_variants,
        skip_agentic=skip_agentic,
    )

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Total records: {metadata.get('total_records', 0)}")
    splits = metadata.get("splits", {})
    for split, count in splits.items():
        console.print(f"  {split}: {count}")

    stats = metadata.get("pipeline_stats", {})
    console.print(f"\n  Gold configs: {stats.get('total_gold', 0)}")
    console.print(f"  Broken configs: {stats.get('total_broken', 0)}")
    console.print(f"  Resistant configs: {stats.get('resistant_configs', 0)}")


@click.command()
@click.option("--gold-dir", default=str(GOLD_DIR), help="Gold configs directory")
@click.option("--output-dir", default=str(TRAINING_DIR), help="Output directory for JSONL")
@click.option("--programmatic-variants", "-p", default=4, help="Programmatic faults per config")
@click.option("--agentic-variants", "-a", default=2, help="Agentic faults per config")
@click.option("--skip-agentic", is_flag=True, help="Skip agentic (LLM) injection")
def main(
    gold_dir: str,
    output_dir: str,
    programmatic_variants: int,
    agentic_variants: int,
    skip_agentic: bool,
):
    """Generate training data pairs from gold IaC configurations."""
    asyncio.run(run_generate(
        gold_dir, output_dir, programmatic_variants, agentic_variants, skip_agentic
    ))


if __name__ == "__main__":
    main()
