"""Output formatter for training data (JSONL with train/val/test splits)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """A single training record (gold + broken pair)."""

    id: str
    format: str  # "terraform" or "cloudformation"
    gold_config: str
    broken_config: str
    errors: list[str]
    warnings: list[str]
    fault_types: list[str]
    fault_description: str
    difficulty: str  # "low", "medium", "high"
    source: str  # "programmatic" or "agentic"
    split: str = ""  # "train", "val", or "test" — assigned by format_and_split
    gold_hash: str = ""  # Hash of gold config for dedup/split

    def __post_init__(self):
        if not self.gold_hash:
            self.gold_hash = hashlib.md5(self.gold_config.encode()).hexdigest()


def format_and_split(
    records: list[TrainingRecord],
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Split records by gold config and write JSONL files.

    Splitting is done by gold config hash (not by record) to prevent
    data leakage — all variants of the same gold config go to the same split.

    Args:
        records: List of TrainingRecord objects.
        output_dir: Directory to write train.jsonl, val.jsonl, test.jsonl.
        ratios: (train, val, test) split ratios.

    Returns:
        Metadata dict with counts and split info.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group records by gold_hash
    gold_groups: dict[str, list[TrainingRecord]] = {}
    for record in records:
        gold_groups.setdefault(record.gold_hash, []).append(record)

    # Deterministic ordering by hash
    sorted_hashes = sorted(gold_groups.keys())
    total_golds = len(sorted_hashes)

    train_end = int(total_golds * ratios[0])
    val_end = train_end + int(total_golds * ratios[1])

    train_hashes = set(sorted_hashes[:train_end])
    val_hashes = set(sorted_hashes[train_end:val_end])
    test_hashes = set(sorted_hashes[val_end:])

    # Assign splits
    splits: dict[str, list[TrainingRecord]] = {"train": [], "val": [], "test": []}
    for h in sorted_hashes:
        if h in train_hashes:
            split = "train"
        elif h in val_hashes:
            split = "val"
        else:
            split = "test"
        for record in gold_groups[h]:
            record.split = split
            splits[split].append(record)

    # Write JSONL files
    counts = {}
    for split_name, split_records in splits.items():
        filepath = output_path / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for record in split_records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        counts[split_name] = len(split_records)
        logger.info("Wrote %d records to %s", len(split_records), filepath)

    # Write metadata
    metadata = {
        "total_records": len(records),
        "total_gold_configs": total_golds,
        "splits": counts,
        "ratios": list(ratios),
        "fault_type_distribution": _count_fault_types(records),
        "format_distribution": _count_formats(records),
        "source_distribution": _count_sources(records),
        "difficulty_distribution": _count_difficulties(records),
    }

    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote metadata to %s", meta_path)

    return metadata


def _count_fault_types(records: list[TrainingRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for ft in r.fault_types:
            counts[ft] = counts.get(ft, 0) + 1
    return dict(sorted(counts.items()))


def _count_formats(records: list[TrainingRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.format] = counts.get(r.format, 0) + 1
    return counts


def _count_sources(records: list[TrainingRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.source] = counts.get(r.source, 0) + 1
    return counts


def _count_difficulties(records: list[TrainingRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.difficulty] = counts.get(r.difficulty, 0) + 1
    return counts
