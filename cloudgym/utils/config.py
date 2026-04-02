"""Project-wide configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
GOLD_TF_DIR = GOLD_DIR / "terraform"
GOLD_CF_DIR = GOLD_DIR / "cloudformation"
BROKEN_DIR = DATA_DIR / "broken"
TRAINING_DIR = DATA_DIR / "training"
BENCHMARK_DIR = DATA_DIR / "benchmark"


@dataclass
class ScraperConfig:
    github_token: str | None = None
    max_repos: int = 200
    min_resources: int = 2
    max_file_size_kb: int = 500
    tf_search_queries: list[str] = field(
        default_factory=lambda: [
            "aws provider terraform",
            "terraform module aws vpc",
            "terraform aws ec2 s3",
            "terraform azurerm resource_group",
            "terraform google_compute_instance",
        ]
    )
    cf_search_queries: list[str] = field(
        default_factory=lambda: [
            "AWSTemplateFormatVersion",
            "cloudformation template aws",
            "cloudformation ec2 vpc",
        ]
    )


@dataclass
class InverterConfig:
    faults_per_config: int = 1
    max_retries: int = 3
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class PipelineConfig:
    programmatic_variants: int = 4
    agentic_variants: int = 2
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
