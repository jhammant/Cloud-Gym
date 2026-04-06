"""Diff and output formatting for repair results."""

from __future__ import annotations

import difflib
from pathlib import Path


def unified_diff(original: str, repaired: str, filename: str = "config") -> str:
    """Generate a unified diff between original and repaired config."""
    original_lines = original.splitlines(keepends=True)
    repaired_lines = repaired.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        repaired_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "".join(diff)


def colorized_diff(original: str, repaired: str, filename: str = "config") -> str:
    """Generate a colorized diff using Rich markup."""
    original_lines = original.splitlines(keepends=True)
    repaired_lines = repaired.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        repaired_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )

    colored_lines = []
    for line in diff:
        line = line.rstrip("\n")
        if line.startswith("+++") or line.startswith("---"):
            colored_lines.append(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            colored_lines.append(f"[cyan]{line}[/cyan]")
        elif line.startswith("+"):
            colored_lines.append(f"[green]{line}[/green]")
        elif line.startswith("-"):
            colored_lines.append(f"[red]{line}[/red]")
        else:
            colored_lines.append(line)

    return "\n".join(colored_lines)


def write_repair(path: Path, content: str) -> None:
    """Write repaired content to file."""
    path.write_text(content)
