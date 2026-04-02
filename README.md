# Cloud-Gym

Scalable Training Data Generation for Infrastructure-as-Code Repair via Environment Inversion.

## Overview

Cloud-Gym generates (broken_config, error_message, fix) training pairs for IaC repair by applying **environment inversion** — taking working Terraform, CloudFormation, and OpenTofu configs and systematically breaking them using a defined fault taxonomy.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# View the fault taxonomy
cloud-gym taxonomy

# Run scrapers to collect gold configs
python scripts/scrape.py

# Run tests
pytest
```

## Project Structure

- `cloudgym/taxonomy/` — Fault type definitions (28+ fault types across 8 categories)
- `cloudgym/scraper/` — Gold config collection from GitHub, Terraform Registry, AWS samples
- `cloudgym/validator/` — IaC validation wrappers (terraform, cfn-lint, tofu)
- `cloudgym/inverter/` — Fault injection engines (programmatic + agentic)
- `cloudgym/generator/` — Training data pipeline and output formatting
- `cloudgym/benchmark/` — Curated benchmark dataset and evaluation harness
