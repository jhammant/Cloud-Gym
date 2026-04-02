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

    def __init__(self, benchmark_path: str | Path):
        self.dataset = BenchmarkDataset(benchmark_path)

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

        raw_results = []

        for entry in self.dataset:
            passes = 0
            for attempt in range(n_attempts):
                try:
                    repaired = await model_fn(entry.broken_config, entry.errors)
                    passed = await self._check_repair(repaired, entry.format)
                except Exception:
                    logger.exception(
                        "Model failed on entry %s attempt %d", entry.id, attempt
                    )
                    passed = False

                if passed:
                    passes += 1

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

    async def _check_repair(self, repaired: str, iac_format: str) -> bool:
        """Check if a repaired config passes validation."""
        if not repaired or not repaired.strip():
            return False

        suffix = ".tf" if iac_format in ("terraform", "opentofu") else ".yaml"
        tmpdir = Path(tempfile.mkdtemp(prefix="cloudgym_eval_"))
        tmp_file = tmpdir / f"repaired{suffix}"
        tmp_file.write_text(repaired)

        try:
            if iac_format in ("terraform", "opentofu"):
                from cloudgym.validator.terraform import validate
            else:
                from cloudgym.validator.cloudformation import validate

            result = await validate(tmp_file)
            return result.valid and len(result.errors) == 0
        except Exception:
            logger.exception("Validation error during eval")
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
