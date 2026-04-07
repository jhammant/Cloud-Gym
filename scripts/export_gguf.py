"""Export fine-tuned MLX LoRA models to GGUF format for cross-platform CPU inference.

Two-step process:
  1. Fuse LoRA adapter into base model + dequantize (mlx_lm fuse)
  2. Convert HuggingFace model to GGUF (llama.cpp convert_hf_to_gguf.py)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()

# Default location of llama.cpp's converter (homebrew install)
DEFAULT_CONVERTER = "/opt/homebrew/Cellar/llama.cpp/8500/bin/convert_hf_to_gguf.py"

PRESETS = {
    "0.5b": {
        "base_model": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "adapter_path": "data/models/iac-repair-adapter-distill-0.5b",
        "output_name": "iac-repair-0.5b.gguf",
    },
    "3b": {
        "base_model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "adapter_path": "data/models/iac-repair-adapter-rank4",
        "output_name": "iac-repair-3b.gguf",
    },
    "3b-rank8": {
        "base_model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "adapter_path": "data/models/iac-repair-adapter",
        "output_name": "iac-repair-3b-rank8.gguf",
    },
    "7b": {
        "base_model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "adapter_path": "data/models/iac-repair-7b-adapter-v2",
        "output_name": "iac-repair-7b.gguf",
    },
}


def _find_converter() -> str:
    """Find the convert_hf_to_gguf.py script."""
    # Check known locations
    for path in [DEFAULT_CONVERTER, shutil.which("convert_hf_to_gguf.py") or ""]:
        if path and Path(path).exists():
            return path

    # Search homebrew
    result = subprocess.run(
        ["find", "/opt/homebrew/Cellar/llama.cpp", "-name", "convert_hf_to_gguf.py"],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        return result.stdout.strip().splitlines()[0]

    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Install llama.cpp: brew install llama.cpp"
    )


@click.command()
@click.argument("preset", type=click.Choice(list(PRESETS.keys())))
@click.option("--output-dir", default="data/models/exports", help="Output directory for GGUF files")
@click.option("--base-model", default=None, help="Override base model path")
@click.option("--adapter-path", default=None, help="Override adapter path")
@click.option("--converter", default=None, help="Path to convert_hf_to_gguf.py")
@click.option("--keep-fused", is_flag=True, help="Keep intermediate fused HF model")
def main(
    preset: str,
    output_dir: str,
    base_model: str | None,
    adapter_path: str | None,
    converter: str | None,
    keep_fused: bool,
):
    """Export a fine-tuned model to GGUF format.

    Two-step process:
      1. Fuse LoRA adapter + dequantize to HuggingFace format (mlx_lm)
      2. Convert HF to GGUF (llama.cpp convert_hf_to_gguf.py)

    Presets: 0.5b (smallest, for Lambda/CI), 3b (best quality), 3b-rank8 (original)
    """
    config = PRESETS[preset]
    model = base_model or config["base_model"]
    adapter = adapter_path or config["adapter_path"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_path = out_dir / f"fused-{preset}"
    gguf_path = out_dir / config["output_name"]

    converter_path = converter or _find_converter()

    console.print(f"\n[bold]Exporting {preset} to GGUF[/bold]")
    console.print(f"  Base model:  {model}")
    console.print(f"  Adapter:     {adapter}")
    console.print(f"  Converter:   {converter_path}")
    console.print(f"  Output:      {gguf_path}\n")

    # Step 1: Fuse adapter + dequantize to HF format
    console.print("[bold green]Step 1:[/bold green] Fuse adapter + dequantize to HuggingFace format...")
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model,
        "--adapter-path", adapter,
        "--save-path", str(fused_path),
        "--dequantize",
    ]
    console.print(f"  [dim]{' '.join(fuse_cmd)}[/dim]\n")

    result = subprocess.run(fuse_cmd, check=False)
    if result.returncode != 0:
        console.print("[red]Fuse step failed![/red]")
        sys.exit(1)

    # Step 2: Convert HF to GGUF
    console.print("\n[bold green]Step 2:[/bold green] Convert to GGUF format...")
    convert_cmd = [
        sys.executable, converter_path,
        str(fused_path),
        "--outfile", str(gguf_path),
        "--outtype", "f16",
    ]
    console.print(f"  [dim]{' '.join(convert_cmd)}[/dim]\n")

    result = subprocess.run(convert_cmd, check=False)
    if result.returncode != 0:
        console.print("[red]GGUF conversion failed![/red]")
        sys.exit(1)

    # Cleanup fused model (large) unless --keep-fused
    if not keep_fused and fused_path.exists():
        console.print(f"[dim]Cleaning up intermediate fused model at {fused_path}[/dim]")
        shutil.rmtree(fused_path)

    if gguf_path.exists():
        size_mb = gguf_path.stat().st_size / (1024 * 1024)
        console.print(f"\n[bold green]Done![/bold green] {gguf_path} ({size_mb:.1f} MB)")
        console.print(f"\nTest with:")
        console.print(f"  iac-fix repair examples/broken_security_group.tf --backend gguf --model {gguf_path}")
    else:
        console.print("[red]GGUF file not found after conversion![/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
