"""Run all scrapers to collect gold IaC configurations."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from cloudgym.scraper.aws_samples import AWSSamplesScraper, save_aws_samples
from cloudgym.scraper.github import GitHubScraper, save_scraped_files
from cloudgym.scraper.registry import RegistryScraper, save_registry_modules
from cloudgym.scraper.validator import validate_all_gold
from cloudgym.utils.config import ScraperConfig

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_scrape(skip_github: bool, skip_registry: bool, skip_aws: bool, skip_validate: bool):
    config = ScraperConfig()

    table = Table(title="Scraping Results")
    table.add_column("Source", style="cyan")
    table.add_column("Files", justify="right", style="green")

    # GitHub
    if not skip_github:
        console.print("[bold]Scraping GitHub...[/bold]")
        gh = GitHubScraper(config=config)
        files = await gh.scrape_all()
        counts = await save_scraped_files(files)
        table.add_row("GitHub (Terraform)", str(counts["terraform"]))
        table.add_row("GitHub (CloudFormation)", str(counts["cloudformation"]))
    else:
        console.print("[dim]Skipping GitHub[/dim]")

    # Terraform Registry
    if not skip_registry:
        console.print("[bold]Scraping Terraform Registry...[/bold]")
        reg = RegistryScraper()
        modules = await reg.scrape()
        count = await save_registry_modules(modules)
        table.add_row("Terraform Registry", str(count))
    else:
        console.print("[dim]Skipping Terraform Registry[/dim]")

    # AWS Samples
    if not skip_aws:
        console.print("[bold]Scraping AWS CF Samples...[/bold]")
        aws = AWSSamplesScraper()
        templates = await aws.scrape()
        count = await save_aws_samples(templates)
        table.add_row("AWS CF Samples", str(count))
    else:
        console.print("[dim]Skipping AWS Samples[/dim]")

    console.print(table)

    # Validate
    if not skip_validate:
        console.print("\n[bold]Validating gold instances...[/bold]")
        stats = await validate_all_gold()

        val_table = Table(title="Validation Results")
        val_table.add_column("Format", style="cyan")
        val_table.add_column("Total", justify="right")
        val_table.add_column("Valid", justify="right", style="green")
        val_table.add_column("Invalid", justify="right", style="red")
        val_table.add_column("Pass Rate", justify="right")

        for fmt, s in stats.items():
            val_table.add_row(fmt, str(s.total), str(s.valid), str(s.invalid), f"{s.pass_rate:.0%}")

        console.print(val_table)


@click.command()
@click.option("--skip-github", is_flag=True, help="Skip GitHub scraping")
@click.option("--skip-registry", is_flag=True, help="Skip Terraform Registry scraping")
@click.option("--skip-aws", is_flag=True, help="Skip AWS samples scraping")
@click.option("--skip-validate", is_flag=True, help="Skip gold validation step")
def main(skip_github: bool, skip_registry: bool, skip_aws: bool, skip_validate: bool):
    """Run all scrapers to collect gold IaC configurations."""
    asyncio.run(run_scrape(skip_github, skip_registry, skip_aws, skip_validate))


if __name__ == "__main__":
    main()
