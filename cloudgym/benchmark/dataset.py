"""Benchmark dataset management.

Curates a balanced subset from the test split for evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """A single benchmark entry."""

    id: str
    format: str
    broken_config: str
    errors: list[str]
    warnings: list[str]
    fault_types: list[str]
    difficulty: str
    gold_config: str
    gold_hash: str


class BenchmarkDataset:
    """Manages the curated benchmark dataset."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: list[BenchmarkEntry] = []
        if self.path.exists():
            self._load()

    def _load(self):
        """Load benchmark entries from JSONL."""
        with open(self.path) as f:
            for line in f:
                data = json.loads(line)
                self.entries.append(BenchmarkEntry(**{
                    k: data[k] for k in BenchmarkEntry.__dataclass_fields__
                    if k in data
                }))
        logger.info("Loaded %d benchmark entries from %s", len(self.entries), self.path)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @staticmethod
    def build(
        test_jsonl: str | Path,
        output_path: str | Path,
        target_size: int = 200,
    ) -> BenchmarkDataset:
        """Curate a benchmark dataset from the test split.

        Curation rules:
        - Single-fault only (one fault type per record)
        - Balanced across categories and difficulties
        - Min 10-line configs (non-trivial)
        - Deduplicated per gold config (max 1 entry per gold hash per fault category)

        Args:
            test_jsonl: Path to test.jsonl from format_and_split.
            output_path: Path to write benchmark.jsonl.
            target_size: Target number of benchmark entries.

        Returns:
            BenchmarkDataset with curated entries.
        """
        test_path = Path(test_jsonl)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load test records
        records = []
        with open(test_path) as f:
            for line in f:
                records.append(json.loads(line))

        logger.info("Loaded %d test records for curation", len(records))

        # Filter: single-fault, min 10 lines
        candidates = [
            r for r in records
            if len(r.get("fault_types", [])) == 1
            and len(r.get("broken_config", "").splitlines()) >= 10
            and r.get("errors")  # Must have validation errors
        ]
        logger.info("%d candidates after filtering", len(candidates))

        # Deduplicate: max 1 entry per (gold_hash, fault_category)
        seen: set[tuple[str, str]] = set()
        deduped = []
        for r in candidates:
            fault_id = r["fault_types"][0]
            category = fault_id.split(".")[0] if "." in fault_id else fault_id
            key = (r.get("gold_hash", ""), category)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        logger.info("%d after deduplication", len(deduped))

        # Balance across categories and difficulties
        selected = _balance_select(deduped, target_size)
        logger.info("Selected %d entries for benchmark", len(selected))

        # Write benchmark JSONL
        with open(out_path, "w") as f:
            for r in selected:
                entry = {
                    "id": r["id"],
                    "format": r["format"],
                    "broken_config": r["broken_config"],
                    "errors": r["errors"],
                    "warnings": r.get("warnings", []),
                    "fault_types": r["fault_types"],
                    "difficulty": r["difficulty"],
                    "gold_config": r["gold_config"],
                    "gold_hash": r.get("gold_hash", ""),
                }
                f.write(json.dumps(entry) + "\n")

        # Write metadata
        meta = {
            "total_entries": len(selected),
            "source": str(test_path),
            "category_distribution": _count_categories(selected),
            "difficulty_distribution": _count_field(selected, "difficulty"),
            "format_distribution": _count_field(selected, "format"),
        }
        meta_path = out_path.parent / "benchmark_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return BenchmarkDataset(out_path)


def _balance_select(records: list[dict], target: int) -> list[dict]:
    """Select records with balanced category/difficulty distribution."""
    by_category: dict[str, list[dict]] = {}
    for r in records:
        fault_id = r["fault_types"][0]
        cat = fault_id.split(".")[0] if "." in fault_id else fault_id
        by_category.setdefault(cat, []).append(r)

    if not by_category:
        return []

    per_category = max(1, target // len(by_category))
    selected = []

    for cat, cat_records in by_category.items():
        # Within category, balance by difficulty
        by_diff: dict[str, list[dict]] = {}
        for r in cat_records:
            by_diff.setdefault(r["difficulty"], []).append(r)

        per_diff = max(1, per_category // max(len(by_diff), 1))
        for diff, diff_records in by_diff.items():
            selected.extend(diff_records[:per_diff])

    return selected[:target]


def _count_categories(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        fault_id = r["fault_types"][0]
        cat = fault_id.split(".")[0] if "." in fault_id else fault_id
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _count_field(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        val = r.get(field, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts
