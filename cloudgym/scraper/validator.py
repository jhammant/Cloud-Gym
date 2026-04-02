"""Gold instance validator — filters scraped configs to only keep valid ones."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cloudgym.utils.config import GOLD_CF_DIR, GOLD_TF_DIR
from cloudgym.validator import cloudformation as cf_validator
from cloudgym.validator import terraform as tf_validator

logger = logging.getLogger(__name__)


@dataclass
class ValidationStats:
    """Statistics from a gold validation run."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    errors: dict[str, list[str]] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.valid / self.total if self.total > 0 else 0.0


async def validate_gold_terraform(
    directory: Path | None = None,
    concurrency: int = 5,
) -> ValidationStats:
    """Validate all Terraform configs in the gold directory.

    Removes configs that fail validation so only gold instances remain.
    """
    tf_dir = directory or GOLD_TF_DIR
    if not tf_dir.exists():
        logger.warning("Terraform gold directory does not exist: %s", tf_dir)
        return ValidationStats()

    tf_files = list(tf_dir.glob("*.tf"))
    stats = ValidationStats(total=len(tf_files))

    semaphore = asyncio.Semaphore(concurrency)

    async def _validate_one(path: Path) -> None:
        async with semaphore:
            result = await tf_validator.validate(path)
            if result.valid:
                stats.valid += 1
                logger.debug("PASS: %s", path.name)
            else:
                stats.invalid += 1
                stats.errors[path.name] = result.errors
                logger.info("FAIL (removing): %s — %s", path.name, result.errors[:2])
                path.unlink(missing_ok=True)

    await asyncio.gather(*[_validate_one(f) for f in tf_files])
    return stats


async def validate_gold_cloudformation(
    directory: Path | None = None,
    concurrency: int = 5,
) -> ValidationStats:
    """Validate all CloudFormation templates in the gold directory.

    Removes templates that fail cfn-lint so only gold instances remain.
    """
    cf_dir = directory or GOLD_CF_DIR
    if not cf_dir.exists():
        logger.warning("CloudFormation gold directory does not exist: %s", cf_dir)
        return ValidationStats()

    cf_files = list(cf_dir.glob("*.yaml")) + list(cf_dir.glob("*.json"))
    stats = ValidationStats(total=len(cf_files))

    semaphore = asyncio.Semaphore(concurrency)

    async def _validate_one(path: Path) -> None:
        async with semaphore:
            result = await cf_validator.validate(path)
            if result.valid:
                stats.valid += 1
                logger.debug("PASS: %s", path.name)
            else:
                stats.invalid += 1
                stats.errors[path.name] = result.errors
                logger.info("FAIL (removing): %s — %s", path.name, result.errors[:2])
                path.unlink(missing_ok=True)

    await asyncio.gather(*[_validate_one(f) for f in cf_files])
    return stats


async def validate_all_gold() -> dict[str, ValidationStats]:
    """Validate all gold configs across formats."""
    tf_stats, cf_stats = await asyncio.gather(
        validate_gold_terraform(),
        validate_gold_cloudformation(),
    )

    logger.info(
        "Gold validation complete — TF: %d/%d pass (%.0f%%), CF: %d/%d pass (%.0f%%)",
        tf_stats.valid,
        tf_stats.total,
        tf_stats.pass_rate * 100,
        cf_stats.valid,
        cf_stats.total,
        cf_stats.pass_rate * 100,
    )

    return {"terraform": tf_stats, "cloudformation": cf_stats}
