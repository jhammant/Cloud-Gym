"""MLX LoRA fine-tuning script for Cloud-Gym IaC repair adapter."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
DEFAULT_DATA = "data/finetune"
DEFAULT_ADAPTER = "data/models/iac-repair-adapter"


@click.command()
@click.option("--model", default=DEFAULT_MODEL, help="Base model to fine-tune")
@click.option("--data", default=DEFAULT_DATA, help="Training data directory")
@click.option("--adapter-path", default=DEFAULT_ADAPTER, help="Output adapter path")
@click.option("--iters", default=1000, type=int, help="Training iterations")
@click.option("--batch-size", default=1, type=int, help="Batch size")
@click.option("--learning-rate", default=2e-5, type=float, help="Learning rate")
def main(
    model: str,
    data: str,
    adapter_path: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
):
    """Run MLX LoRA fine-tuning for IaC repair."""
    # Check that mlx-lm is installed
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        console.print("[red]mlx-lm is not installed. Run: pip install mlx-lm[/red]")
        sys.exit(1)

    # Load adapter config for LoRA-specific parameters
    config_path = Path(adapter_path) / "adapter_config.json"
    config: dict = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
        console.print(f"[dim]Loaded adapter config from {config_path}[/dim]")

    lora_params = config.get("lora_parameters", {})

    # Auto-detect latest checkpoint for resume
    adapter_dir = Path(adapter_path)
    checkpoints = sorted(adapter_dir.glob("0*_adapters.safetensors"))
    resume_file = str(checkpoints[-1]) if checkpoints else None
    if resume_file:
        console.print(f"[yellow]Resuming from checkpoint: {resume_file}[/yellow]")

    # Write a runtime config merging adapter config with CLI overrides
    # mlx_lm.lora reads LoRA params (rank, scale, dropout) from the config file
    run_config = dict(config)
    run_config.update({
        "model": model,
        "data": data,
        "adapter_path": adapter_path,
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train": True,
        "resume_adapter_file": resume_file,
    })
    run_config_path = Path(adapter_path) / "train_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path.write_text(json.dumps(run_config, indent=2))

    # Build the mlx_lm.lora command using config file
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "-c", str(run_config_path),
    ]

    console.print(f"\n[bold green]Starting LoRA fine-tuning[/bold green]")
    console.print(f"  Model:         {model}")
    console.print(f"  Data:          {data}")
    console.print(f"  Adapter:       {adapter_path}")
    console.print(f"  Iterations:    {iters}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  LoRA rank:     {lora_params.get('rank', 8)}")
    console.print()

    subprocess.run(cmd, check=True)

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Adapter saved to [cyan]{adapter_path}[/cyan]")


if __name__ == "__main__":
    main()
