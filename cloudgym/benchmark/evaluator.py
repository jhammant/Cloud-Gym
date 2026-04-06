"""Evaluation harness for IaC repair benchmark.

Validates model-generated fixes via terraform validate / cfn-lint
and computes pass@k metrics using the unbiased Codex estimator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from cloudgym.benchmark.dataset import BenchmarkDataset, BenchmarkEntry

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Evaluation report for a model on the benchmark."""

    model_name: str
    n_attempts: int
    total_entries: int
    pass_at_k: dict[int, float] = field(default_factory=dict)
    per_category: dict[str, dict[int, float]] = field(default_factory=dict)
    per_difficulty: dict[str, dict[int, float]] = field(default_factory=dict)
    per_format: dict[str, dict[int, float]] = field(default_factory=dict)
    raw_results: list[dict] = field(default_factory=list)


# Type alias for model repair function
ModelFn = Callable[[str, list[str]], Awaitable[str]]


class Evaluator:
    """Evaluates model repair attempts against the benchmark."""

    # Max concurrent validations (terraform/cfn-lint)
    DEFAULT_CONCURRENCY = 8

    def __init__(self, benchmark_path: str | Path, concurrency: int | None = None):
        self.dataset = BenchmarkDataset(benchmark_path)
        self._tf_cache_dir: Path | None = None
        self._concurrency = concurrency or self.DEFAULT_CONCURRENCY

    async def evaluate_model(
        self,
        model_fn: ModelFn,
        model_name: str = "unknown",
        n_attempts: int = 5,
        k_values: list[int] | None = None,
    ) -> EvalReport:
        """Evaluate a model's repair ability on the benchmark.

        Args:
            model_fn: Async function (broken_config, errors) -> repaired_config.
            model_name: Name of the model being evaluated.
            n_attempts: Number of repair attempts per benchmark entry.
            k_values: k values for pass@k computation. Default [1, 3].

        Returns:
            EvalReport with pass@k metrics and breakdowns.
        """
        if k_values is None:
            k_values = [1, 3]

        # Phase 1: Generate all repairs (serial — model inference isn't parallel-safe)
        # Store as list of (entry, [repaired_configs])
        all_repairs: list[tuple[BenchmarkEntry, list[str | None]]] = []
        for entry in self.dataset:
            repairs: list[str | None] = []
            for attempt in range(n_attempts):
                try:
                    repaired = await model_fn(entry.broken_config, entry.errors)
                    repairs.append(repaired)
                except Exception:
                    logger.exception(
                        "Model failed on entry %s attempt %d", entry.id, attempt
                    )
                    repairs.append(None)
            all_repairs.append((entry, repairs))

        # Phase 2: Validate all repairs concurrently
        sem = asyncio.Semaphore(self._concurrency)

        async def _validate(repaired: str | None, fmt: str) -> bool:
            if repaired is None:
                return False
            async with sem:
                return await self._check_repair(repaired, fmt)

        raw_results = []
        for entry, repairs in all_repairs:
            tasks = [_validate(r, entry.format) for r in repairs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            passes = sum(
                1 for r in results if r is True
            )
            raw_results.append({
                "id": entry.id,
                "format": entry.format,
                "fault_types": entry.fault_types,
                "difficulty": entry.difficulty,
                "n": n_attempts,
                "c": passes,
            })

        # Compute metrics
        report = EvalReport(
            model_name=model_name,
            n_attempts=n_attempts,
            total_entries=len(self.dataset),
            raw_results=raw_results,
        )

        # Overall pass@k
        for k in k_values:
            report.pass_at_k[k] = _compute_pass_at_k(raw_results, k)

        # Per-category breakdown
        categories = set()
        for r in raw_results:
            for ft in r["fault_types"]:
                cat = ft.split(".")[0] if "." in ft else ft
                categories.add(cat)

        for cat in categories:
            cat_results = [
                r for r in raw_results
                if any(ft.startswith(cat) for ft in r["fault_types"])
            ]
            report.per_category[cat] = {
                k: _compute_pass_at_k(cat_results, k) for k in k_values
            }

        # Per-difficulty breakdown
        difficulties = {r["difficulty"] for r in raw_results}
        for diff in difficulties:
            diff_results = [r for r in raw_results if r["difficulty"] == diff]
            report.per_difficulty[diff] = {
                k: _compute_pass_at_k(diff_results, k) for k in k_values
            }

        # Per-format breakdown
        formats = {r["format"] for r in raw_results}
        for fmt in formats:
            fmt_results = [r for r in raw_results if r["format"] == fmt]
            report.per_format[fmt] = {
                k: _compute_pass_at_k(fmt_results, k) for k in k_values
            }

        return report

    async def _ensure_tf_cache(self) -> Path:
        """Create a cached terraform init directory for fast validation."""
        if self._tf_cache_dir and self._tf_cache_dir.exists():
            return self._tf_cache_dir

        cache = Path(tempfile.mkdtemp(prefix="cloudgym_tf_cache_"))
        # Write a minimal .tf that requires the AWS provider
        (cache / "providers.tf").write_text(
            'terraform {\n  required_providers {\n'
            '    aws = {\n      source  = "hashicorp/aws"\n'
            '      version = "~> 5.0"\n    }\n  }\n}\n'
        )
        import asyncio
        proc = await asyncio.create_subprocess_exec(
            "terraform", "init", "-backend=false", "-no-color",
            cwd=cache,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        # Remove the providers.tf so it doesn't interfere with validation
        (cache / "providers.tf").unlink(missing_ok=True)
        self._tf_cache_dir = cache
        logger.info("Cached terraform init at %s", cache)
        return cache

    async def _check_repair(self, repaired: str, iac_format: str) -> bool:
        """Check if a repaired config passes validation."""
        if not repaired or not repaired.strip():
            return False

        if iac_format in ("terraform", "opentofu"):
            return await self._check_terraform_repair(repaired)
        else:
            return await self._check_cf_repair(repaired)

    async def _check_terraform_repair(self, repaired: str) -> bool:
        """Validate terraform repair using cached init directory."""
        try:
            cache = await self._ensure_tf_cache()
            tmpdir = Path(tempfile.mkdtemp(prefix="cloudgym_eval_"))
            # Symlink .terraform and lock file from cache (much faster than copy)
            tf_dir = cache / ".terraform"
            lock_file = cache / ".terraform.lock.hcl"
            if tf_dir.exists():
                (tmpdir / ".terraform").symlink_to(tf_dir)
            if lock_file.exists():
                (tmpdir / ".terraform.lock.hcl").symlink_to(lock_file)

            (tmpdir / "repaired.tf").write_text(repaired)

            # Skip init, go straight to validate
            proc = await asyncio.create_subprocess_exec(
                "terraform", "validate", "-json", "-no-color",
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            result = json.loads(stdout.decode(errors="replace"))
            return result.get("valid", False)
        except Exception:
            logger.exception("Terraform validation error during eval")
            return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def _check_cf_repair(self, repaired: str) -> bool:
        """Validate CloudFormation repair via cfn-lint."""
        tmpdir = Path(tempfile.mkdtemp(prefix="cloudgym_eval_"))
        tmp_file = tmpdir / "repaired.yaml"
        tmp_file.write_text(repaired)
        try:
            from cloudgym.validator.cloudformation import validate
            result = await validate(tmp_file)
            return result.valid and len(result.errors) == 0
        except Exception:
            logger.exception("CF validation error during eval")
            return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _compute_pass_at_k(results: list[dict], k: int) -> float:
    """Compute pass@k using the unbiased Codex estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Where n = total attempts, c = number of correct attempts.
    """
    if not results:
        return 0.0

    total = 0.0
    for r in results:
        n = r["n"]
        c = r["c"]
        if n < k:
            # Not enough attempts
            total += 1.0 if c > 0 else 0.0
        elif c == 0:
            total += 0.0
        elif n - c < k:
            total += 1.0
        else:
            total += 1.0 - _comb(n - c, k) / _comb(n, k)

    return total / len(results)


def _comb(n: int, k: int) -> float:
    """Compute combination C(n, k) using math.comb."""
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))
