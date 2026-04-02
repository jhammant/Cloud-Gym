"""End-to-end training data generation pipeline.

Discovers gold configs, runs fault injection (programmatic and/or agentic),
validates breaks, and outputs training records as JSONL.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from cloudgym.generator.formatter import TrainingRecord, format_and_split
from cloudgym.inverter.engine import InversionEngine, InversionResult
from cloudgym.taxonomy import REGISTRY
from cloudgym.taxonomy.base import FaultCategory, FaultType, IaCFormat
from cloudgym.utils.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""

    total_gold: int = 0
    total_broken: int = 0
    resistant_configs: int = 0
    faults_not_applicable: dict[str, int] = field(default_factory=dict)
    errors: int = 0


class PipelineRunner:
    """Runs the full training data generation pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        max_retries: int = 3,
        concurrency: int = 5,
        skip_validation: bool = True,
    ):
        self.config = config or PipelineConfig()
        self.engine = InversionEngine(
            max_retries=max_retries,
            concurrency=concurrency,
            skip_validation=skip_validation,
        )
        self.stats = PipelineStats()

    async def run(
        self,
        gold_dir: str | Path,
        output_dir: str | Path,
        programmatic_variants: int | None = None,
        agentic_variants: int | None = None,
        skip_agentic: bool = False,
    ) -> dict:
        """Run the full pipeline: discover gold -> inject faults -> write JSONL.

        Args:
            gold_dir: Directory containing gold configs.
            output_dir: Directory to write JSONL output files.
            programmatic_variants: Number of programmatic faults per gold config.
            agentic_variants: Number of agentic faults per gold config.
            skip_agentic: Skip agentic injection entirely.

        Returns:
            Metadata dict from format_and_split.
        """
        gold_path = Path(gold_dir)
        n_prog = programmatic_variants or self.config.programmatic_variants
        n_agent = 0 if skip_agentic else (agentic_variants or self.config.agentic_variants)

        # Discover gold configs
        gold_files = self._discover_gold(gold_path)
        self.stats.total_gold = len(gold_files)
        logger.info("Discovered %d gold configs in %s", len(gold_files), gold_path)

        if not gold_files:
            logger.warning("No gold configs found")
            return {"total_records": 0}

        # Generate records
        records: list[TrainingRecord] = []

        for gold_file in gold_files:
            iac_format = self._detect_format(gold_file)
            applicable_faults = self._get_stratified_faults(iac_format, n_prog)

            # Programmatic inversions
            for fault_type in applicable_faults:
                result = await self.engine.invert(
                    gold_file, [fault_type], mode="programmatic"
                )
                if result is not None:
                    record = self._result_to_record(result, "programmatic")
                    records.append(record)
                    self.stats.total_broken += 1
                else:
                    key = fault_type.id
                    self.stats.faults_not_applicable[key] = (
                        self.stats.faults_not_applicable.get(key, 0) + 1
                    )

            # Agentic inversions
            if n_agent > 0:
                categories = list({ft.category.name for ft in applicable_faults})
                for i in range(min(n_agent, len(categories))):
                    cat = categories[i % len(categories)]
                    cat_faults = [ft for ft in applicable_faults if ft.category.name == cat]
                    if cat_faults:
                        result = await self.engine.invert(
                            gold_file, cat_faults[:1], mode="agentic"
                        )
                        if result is not None:
                            record = self._result_to_record(result, "agentic")
                            records.append(record)
                            self.stats.total_broken += 1

            if not any(
                r for r in records
                if r.gold_hash == TrainingRecord(
                    id="", format="", gold_config=gold_file.read_text(),
                    broken_config="", errors=[], warnings=[], fault_types=[],
                    fault_description="", difficulty="", source=""
                ).gold_hash
            ):
                self.stats.resistant_configs += 1

        logger.info(
            "Pipeline complete: %d gold -> %d broken (%d resistant)",
            self.stats.total_gold, self.stats.total_broken, self.stats.resistant_configs,
        )

        # Format and split
        if records:
            metadata = format_and_split(
                records,
                output_dir,
                ratios=(
                    self.config.train_split,
                    self.config.val_split,
                    self.config.test_split,
                ),
            )
        else:
            metadata = {"total_records": 0}

        metadata["pipeline_stats"] = {
            "total_gold": self.stats.total_gold,
            "total_broken": self.stats.total_broken,
            "resistant_configs": self.stats.resistant_configs,
            "faults_not_applicable": self.stats.faults_not_applicable,
            "errors": self.stats.errors,
        }

        return metadata

    def _discover_gold(self, gold_dir: Path) -> list[Path]:
        """Find all gold config files."""
        extensions = {".tf", ".yaml", ".yml", ".json", ".template"}
        files = []
        if gold_dir.exists():
            for ext in extensions:
                files.extend(gold_dir.rglob(f"*{ext}"))
        # Filter out hidden files and common non-config files
        files = [
            f for f in files
            if not any(part.startswith('.') for part in f.parts)
            and f.stat().st_size > 50  # Skip trivially small files
        ]
        return sorted(files)

    def _detect_format(self, path: Path) -> str:
        """Detect IaC format from file path."""
        if path.suffix == ".tf":
            return "terraform"
        if "cloudformation" in str(path).lower() or path.suffix in (".yaml", ".yml", ".json", ".template"):
            # Check if it's in a cloudformation directory
            if "cloudformation" in str(path).lower():
                return "cloudformation"
            # Check if it looks like a CF template
            try:
                import yaml
                content = yaml.safe_load(path.read_text())
                if isinstance(content, dict) and ("AWSTemplateFormatVersion" in content or "Resources" in content):
                    return "cloudformation"
            except Exception:
                pass
        return "terraform"

    def _get_stratified_faults(self, iac_format: str, n: int) -> list[FaultType]:
        """Select N fault types via round-robin across categories for balance."""
        fmt_map = {
            "terraform": IaCFormat.TERRAFORM,
            "opentofu": IaCFormat.OPENTOFU,
            "cloudformation": IaCFormat.CLOUDFORMATION,
        }
        iac_fmt = fmt_map.get(iac_format)
        if iac_fmt is None:
            return []

        all_faults = REGISTRY.list_by_format(iac_fmt)
        if not all_faults:
            return []

        # Group by category
        by_category: dict[str, list[FaultType]] = {}
        for ft in all_faults:
            by_category.setdefault(ft.category.name, []).append(ft)

        # Round-robin across categories
        selected: list[FaultType] = []
        categories = list(by_category.keys())
        cat_indices = {cat: 0 for cat in categories}

        while len(selected) < n and len(selected) < len(all_faults):
            for cat in categories:
                if len(selected) >= n:
                    break
                faults = by_category[cat]
                idx = cat_indices[cat]
                if idx < len(faults):
                    selected.append(faults[idx])
                    cat_indices[cat] = idx + 1

            # Check if we've exhausted all categories
            if all(cat_indices[c] >= len(by_category[c]) for c in categories):
                break

        return selected

    def _result_to_record(
        self, result: InversionResult, source: str
    ) -> TrainingRecord:
        """Convert an InversionResult to a TrainingRecord."""
        return TrainingRecord(
            id=str(uuid.uuid4()),
            format=result.iac_format,
            gold_config=result.gold_config,
            broken_config=result.broken_config,
            errors=result.validation_result.errors,
            warnings=result.validation_result.warnings,
            fault_types=[result.fault_type.id],
            fault_description=result.injection.description,
            difficulty=result.fault_type.severity.value,
            source=source,
        )


# Convenience function matching original stub
async def generate_training_data(
    gold_dir: str,
    output_dir: str,
    programmatic_variants: int = 4,
    agentic_variants: int = 2,
) -> dict:
    """Generate training pairs from gold configs.

    Returns statistics about the generation run.
    """
    runner = PipelineRunner()
    return await runner.run(
        gold_dir=gold_dir,
        output_dir=output_dir,
        programmatic_variants=programmatic_variants,
        agentic_variants=agentic_variants,
    )
