"""Cloud-Gym CLI entry point."""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="cloud-gym")
def main():
    """Cloud-Gym: IaC Repair Benchmark via Environment Inversion."""


@main.command()
@click.option("--skip-github", is_flag=True)
@click.option("--skip-registry", is_flag=True)
@click.option("--skip-aws", is_flag=True)
@click.option("--skip-validate", is_flag=True)
def scrape(skip_github: bool, skip_registry: bool, skip_aws: bool, skip_validate: bool):
    """Collect gold IaC configurations from various sources."""
    import asyncio
    from scripts.scrape import run_scrape

    asyncio.run(run_scrape(skip_github, skip_registry, skip_aws, skip_validate))


@main.command()
def taxonomy():
    """Display the fault taxonomy."""
    from rich.console import Console
    from rich.table import Table

    from cloudgym.taxonomy.base import REGISTRY, FaultCategory, IaCFormat
    import cloudgym.taxonomy.terraform  # noqa: F401 — triggers registration
    import cloudgym.taxonomy.cloudformation  # noqa: F401

    console = Console()
    table = Table(title="Cloud-Gym Fault Taxonomy")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Severity")
    table.add_column("Formats")
    table.add_column("Description")

    for fault in sorted(REGISTRY.all(), key=lambda f: (f.category.name, f.severity.value)):
        formats = ", ".join(f.value for f in fault.applicable_formats)
        sev_style = {"low": "green", "medium": "yellow", "high": "red"}[fault.severity.value]
        table.add_row(
            fault.id,
            fault.category.name,
            f"[{sev_style}]{fault.severity.value}[/{sev_style}]",
            formats,
            fault.description,
        )

    console.print(table)
    console.print(f"\nTotal fault types: {len(REGISTRY)}")
    for cat in FaultCategory:
        count = len(REGISTRY.list_by_category(cat))
        if count > 0:
            console.print(f"  {cat.name}: {count}")
