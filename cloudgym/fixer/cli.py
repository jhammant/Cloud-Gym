"""stackfix CLI: AI-powered Infrastructure-as-Code repair tool.

Usage:
    stackfix check main.tf              # validate and show errors
    stackfix repair main.tf             # validate, fix, show diff
    stackfix repair main.tf --apply     # validate, fix, write in place
    stackfix repair main.tf --apply -o fixed.tf  # write to different file
    stackfix repair *.tf                # fix multiple files
    cat broken.tf | stackfix repair -   # stdin/stdout mode
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cloudgym.fixer.detector import IaCFormat, detect_format, validate_file_sync
from cloudgym.fixer.formatter import colorized_diff, unified_diff, write_repair

console = Console()
stderr_console = Console(stderr=True)

# Lazy-loaded repairer (avoids model load until needed)
_repairer = None


def _get_repairer(backend: str, model: str | None, adapter: str | None):
    """Get or create the repairer instance."""
    global _repairer
    if _repairer is not None:
        return _repairer

    if backend == "mlx":
        from cloudgym.fixer.repairer import MLXRepairer, DEFAULT_BASE_MODEL, DEFAULT_ADAPTER_PATH

        _repairer = MLXRepairer(
            base_model=model or DEFAULT_BASE_MODEL,
            adapter_path=adapter or DEFAULT_ADAPTER_PATH,
        )
    elif backend == "gguf":
        from cloudgym.fixer.repairer import GGUFRepairer

        if not model:
            raise click.ClickException(
                "GGUF backend requires --model pointing to a .gguf file. "
                "Export one with: python scripts/export_gguf.py 0.5b"
            )
        _repairer = GGUFRepairer(model_path=model)
    elif backend == "ollama":
        from cloudgym.fixer.repairer import OllamaRepairer

        _repairer = OllamaRepairer(model=model or "qwen2.5-coder:3b")
    else:
        raise click.ClickException(f"Unknown backend: {backend}")

    return _repairer


@click.group()
@click.version_option(version="0.1.1", prog_name="stackfix")
def cli():
    """AI-powered Infrastructure-as-Code repair.

    Validates Terraform, CloudFormation, and OpenTofu configs,
    then uses a fine-tuned local model to fix errors.
    """


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path())
@click.option("--format", "fmt", type=click.Choice(["terraform", "cloudformation", "opentofu"]),
              default=None, help="Override auto-detected format")
def check(files: tuple[str, ...], fmt: str | None):
    """Validate IaC files and report errors.

    Examples:
        stackfix check main.tf
        stackfix check *.yaml
        stackfix check --format cloudformation template.yaml
    """
    iac_fmt = IaCFormat(fmt) if fmt else None
    any_errors = False

    for file_str in files:
        path = Path(file_str)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            any_errors = True
            continue

        detected_fmt, result = validate_file_sync(path, iac_fmt)

        if result.valid:
            console.print(f"[green]PASS[/green] {path} ({detected_fmt.value})")
        else:
            any_errors = True
            console.print(f"[red]FAIL[/red] {path} ({detected_fmt.value})")
            for err in result.errors:
                console.print(f"  [red]error:[/red] {err}")
            for warn in result.warnings:
                console.print(f"  [yellow]warning:[/yellow] {warn}")

    sys.exit(1 if any_errors else 0)


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(allow_dash=True))
@click.option("--apply", is_flag=True, help="Write fixes in place (otherwise just show diff)")
@click.option("-o", "--output", type=click.Path(), help="Write fixed output to this file")
@click.option("--format", "fmt", type=click.Choice(["terraform", "cloudformation", "opentofu"]),
              default=None, help="Override auto-detected format")
@click.option("--backend", type=click.Choice(["mlx", "gguf", "ollama"]), default="mlx",
              help="Model backend (default: mlx)")
@click.option("--model", default=None, help="Override base model")
@click.option("--adapter", default=None, help="Override adapter path")
@click.option("--no-verify", is_flag=True, help="Skip post-repair validation")
@click.option("--diff/--no-diff", default=True, help="Show diff output (default: on)")
@click.option("--color/--no-color", default=True, help="Colorize diff output")
def repair(
    files: tuple[str, ...],
    apply: bool,
    output: str | None,
    fmt: str | None,
    backend: str,
    model: str | None,
    adapter: str | None,
    no_verify: bool,
    diff: bool,
    color: bool,
):
    """Validate and repair IaC files using a fine-tuned AI model.

    By default shows a diff of proposed changes. Use --apply to write fixes.

    Examples:
        stackfix repair main.tf                    # show diff
        stackfix repair main.tf --apply            # fix in place
        stackfix repair main.tf -o fixed.tf        # write to new file
        stackfix repair --backend ollama main.tf   # use Ollama
        cat broken.tf | stackfix repair -          # stdin/stdout
    """
    iac_fmt = IaCFormat(fmt) if fmt else None
    stdin_mode = len(files) == 1 and files[0] == "-"

    if stdin_mode:
        _repair_stdin(iac_fmt, backend, model, adapter, no_verify, output)
        return

    any_failed = False
    for file_str in files:
        path = Path(file_str)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            any_failed = True
            continue

        success = _repair_file(
            path, iac_fmt, backend, model, adapter,
            apply, output, no_verify, diff, color,
        )
        if not success:
            any_failed = True

    sys.exit(1 if any_failed else 0)


def _repair_file(
    path: Path,
    fmt: IaCFormat | None,
    backend: str,
    model: str | None,
    adapter: str | None,
    apply: bool,
    output: str | None,
    no_verify: bool,
    show_diff: bool,
    color: bool,
) -> bool:
    """Repair a single file. Returns True if successful."""
    original = path.read_text()

    # Step 1: Validate
    detected_fmt, result = validate_file_sync(path, fmt)

    if result.valid:
        console.print(f"[green]PASS[/green] {path} — no errors to fix")
        return True

    console.print(f"[yellow]FIXING[/yellow] {path} ({detected_fmt.value}) — {len(result.errors)} error(s)")
    for err in result.errors:
        console.print(f"  [dim]{err}[/dim]")

    # Step 2: Repair
    repairer = _get_repairer(backend, model, adapter)
    with console.status("[bold]Generating fix...", spinner="dots"):
        repaired = repairer.repair(original, result.errors)

    if not repaired or repaired.strip() == original.strip():
        console.print(f"  [yellow]No changes generated[/yellow]")
        return False

    # Step 3: Post-repair validation
    if not no_verify:
        verified = _verify_repair(repaired, detected_fmt, path.name)
        if not verified:
            console.print(f"  [red]Fix did not pass validation — not applying[/red]")
            if show_diff:
                _show_diff(original, repaired, path.name, color)
            return False

    # Step 4: Show diff
    if show_diff:
        _show_diff(original, repaired, path.name, color)

    # Step 5: Apply
    if apply:
        out_path = Path(output) if output else path
        write_repair(out_path, repaired)
        console.print(f"  [green]Fixed and saved to {out_path}[/green]")
    elif output:
        write_repair(Path(output), repaired)
        console.print(f"  [green]Fixed output written to {output}[/green]")
    elif not apply:
        console.print(f"  [dim]Use --apply to write fix, or -o FILE to save elsewhere[/dim]")

    return True


def _repair_stdin(
    fmt: IaCFormat | None,
    backend: str,
    model: str | None,
    adapter: str | None,
    no_verify: bool,
    output: str | None,
):
    """Repair from stdin, output to stdout or file."""
    original = sys.stdin.read()

    # Write to temp file for validation
    suffix = ".tf" if fmt != IaCFormat.CLOUDFORMATION else ".yaml"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(original)
        tmp_path = Path(f.name)

    try:
        detected_fmt, result = validate_file_sync(tmp_path, fmt)
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.valid:
        # Pass through unchanged
        sys.stdout.write(original)
        return

    stderr_console.print(f"[yellow]Fixing stdin[/yellow] ({detected_fmt.value}) — {len(result.errors)} error(s)")

    repairer = _get_repairer(backend, model, adapter)
    repaired = repairer.repair(original, result.errors)

    if output:
        write_repair(Path(output), repaired)
        stderr_console.print(f"[green]Written to {output}[/green]")
    else:
        sys.stdout.write(repaired)


def _verify_repair(repaired: str, fmt: IaCFormat, filename: str) -> bool:
    """Validate repaired content to confirm the fix works."""
    suffix = ".tf" if fmt != IaCFormat.CLOUDFORMATION else ".yaml"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(repaired)
        tmp_path = Path(f.name)

    try:
        _, result = validate_file_sync(tmp_path, fmt)
        if result.valid:
            console.print(f"  [green]Verified: fix passes validation[/green]")
        else:
            console.print(f"  [red]Verification failed: {len(result.errors)} error(s) remain[/red]")
            for err in result.errors:
                console.print(f"    [dim]{err}[/dim]")
        return result.valid
    finally:
        tmp_path.unlink(missing_ok=True)


def _show_diff(original: str, repaired: str, filename: str, color: bool):
    """Display diff between original and repaired."""
    if color:
        diff_text = colorized_diff(original, repaired, filename)
        if diff_text:
            console.print(Panel(diff_text, title="Proposed Fix", border_style="blue"))
    else:
        diff_text = unified_diff(original, repaired, filename)
        if diff_text:
            click.echo(diff_text)


@cli.command(name="pre-commit")
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--backend", type=click.Choice(["mlx", "gguf", "ollama"]), default="mlx")
@click.option("--model", default=None)
@click.option("--adapter", default=None)
def pre_commit(files: tuple[str, ...], backend: str, model: str | None, adapter: str | None):
    """Pre-commit hook: validate and auto-fix staged IaC files.

    Exits 0 if all files are valid (or were successfully fixed).
    Exits 1 if any file has unfixable errors.

    Usage in .pre-commit-config.yaml:
        - repo: local
          hooks:
            - id: stackfix
              name: stackfix
              entry: stackfix pre-commit
              language: python
              types_or: [terraform, yaml]
    """
    if not files:
        sys.exit(0)

    iac_files = [
        Path(f) for f in files
        if f.endswith((".tf", ".yaml", ".yml", ".json"))
    ]

    if not iac_files:
        sys.exit(0)

    any_failed = False
    any_fixed = False

    for path in iac_files:
        detected_fmt, result = validate_file_sync(path)

        if result.valid:
            continue

        console.print(f"[yellow]stackfix:[/yellow] {path} has {len(result.errors)} error(s), attempting fix...")

        repairer = _get_repairer(backend, model, adapter)
        original = path.read_text()
        repaired = repairer.repair(original, result.errors)

        if not repaired or repaired.strip() == original.strip():
            console.print(f"  [red]Could not fix {path}[/red]")
            any_failed = True
            continue

        # Verify the fix
        verified = _verify_repair(repaired, detected_fmt, path.name)
        if verified:
            write_repair(path, repaired)
            console.print(f"  [green]Fixed {path}[/green]")
            any_fixed = True
        else:
            console.print(f"  [red]Fix for {path} did not pass validation[/red]")
            any_failed = True

    if any_fixed:
        console.print("[yellow]stackfix: Files were modified. Please re-stage and commit.[/yellow]")
        sys.exit(1)  # Signal to pre-commit that files changed

    sys.exit(1 if any_failed else 0)


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["terraform", "cloudformation", "opentofu"]),
              default=None, help="Override auto-detected format")
@click.option("--backend", type=click.Choice(["mlx", "gguf", "ollama"]), default="mlx")
@click.option("--model", default=None)
@click.option("--adapter", default=None)
def discuss(files: tuple[str, ...], fmt: str | None, backend: str, model: str | None, adapter: str | None):
    """Explain IaC errors in plain language with fix guidance.

    Uses the fine-tuned model to analyze errors, explain what's wrong,
    why it matters, and how to fix it. Great for learning and code review.

    Examples:
        stackfix discuss main.tf
        stackfix discuss --backend ollama template.yaml
    """
    iac_fmt = IaCFormat(fmt) if fmt else None

    for file_str in files:
        path = Path(file_str)
        config = path.read_text()
        detected_fmt, result = validate_file_sync(path, iac_fmt)

        if result.valid and not result.warnings:
            console.print(f"[green]PASS[/green] {path} — no issues to discuss")
            continue

        errors = result.errors + result.warnings
        console.print(f"\n[bold]{path}[/bold] ({detected_fmt.value})")
        console.print(f"[dim]{len(result.errors)} error(s), {len(result.warnings)} warning(s)[/dim]\n")

        repairer = _get_repairer(backend, model, adapter)
        with console.status("[bold]Analyzing...", spinner="dots"):
            explanation = repairer.discuss(config, errors)

        console.print(Panel(explanation, title="Analysis", border_style="blue"))


@cli.command(name="git-diff")
@click.option("--backend", type=click.Choice(["mlx", "gguf", "ollama"]), default="mlx")
@click.option("--model", default=None)
@click.option("--adapter", default=None)
@click.option("--apply", is_flag=True, help="Auto-fix and stage repaired files")
def git_diff(backend: str, model: str | None, adapter: str | None, apply: bool):
    """Check and repair IaC files changed in the current git diff.

    Scans staged and unstaged IaC file changes, validates them,
    and offers AI-powered fixes. Ideal for CI or local git workflows.

    Examples:
        stackfix git-diff                 # check changed IaC files
        stackfix git-diff --apply         # fix and re-stage
    """
    import subprocess

    # Get list of changed IaC files (staged + unstaged)
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        changed = result.stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        # No HEAD yet (initial commit) — check staged files
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                capture_output=True, text=True, check=True,
            )
            changed = result.stdout.strip().splitlines()
        except subprocess.CalledProcessError:
            console.print("[red]Not a git repository or git not available[/red]")
            sys.exit(1)

    # Also include unstaged changes
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR"],
            capture_output=True, text=True, check=True,
        )
        changed += result.stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        pass

    # Deduplicate and filter to IaC files
    iac_files = sorted(set(
        f for f in changed
        if f.endswith((".tf", ".yaml", ".yml", ".json"))
        and Path(f).exists()
    ))

    if not iac_files:
        console.print("[green]No changed IaC files to check[/green]")
        sys.exit(0)

    console.print(f"[bold]Checking {len(iac_files)} changed IaC file(s)...[/bold]")

    any_errors = False
    any_fixed = False

    for file_str in iac_files:
        path = Path(file_str)
        detected_fmt, result = validate_file_sync(path)

        if result.valid:
            console.print(f"  [green]PASS[/green] {path}")
            continue

        console.print(f"  [red]FAIL[/red] {path} — {len(result.errors)} error(s)")
        for err in result.errors:
            console.print(f"    [dim]{err}[/dim]")

        if apply:
            repairer = _get_repairer(backend, model, adapter)
            original = path.read_text()
            with console.status(f"  Fixing {path}...", spinner="dots"):
                repaired = repairer.repair(original, result.errors)

            if repaired and repaired.strip() != original.strip():
                verified = _verify_repair(repaired, detected_fmt, path.name)
                if verified:
                    write_repair(path, repaired)
                    subprocess.run(["git", "add", str(path)], check=True)
                    console.print(f"    [green]Fixed and staged[/green]")
                    any_fixed = True
                else:
                    any_errors = True
            else:
                any_errors = True
        else:
            any_errors = True

    if any_fixed:
        console.print(f"\n[green]{sum(1 for _ in iac_files)} file(s) checked, fixes applied and staged[/green]")
    elif any_errors:
        console.print(f"\n[yellow]Use --apply to auto-fix errors[/yellow]")

    sys.exit(1 if any_errors and not apply else 0)


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
