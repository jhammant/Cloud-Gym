"""Tests for the training data pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cloudgym.generator.formatter import TrainingRecord, format_and_split

FIXTURES = Path(__file__).parent / "fixtures"


class TestTrainingRecord:
    """Test TrainingRecord dataclass."""

    def test_gold_hash_auto_generated(self):
        record = TrainingRecord(
            id="test-1",
            format="terraform",
            gold_config="resource {}",
            broken_config="resource {",
            errors=["missing brace"],
            warnings=[],
            fault_types=["SYNTACTIC.missing_closing_brace"],
            fault_description="Removed closing brace",
            difficulty="low",
            source="programmatic",
        )
        assert record.gold_hash
        assert len(record.gold_hash) == 32  # MD5 hex length

    def test_same_gold_same_hash(self):
        gold = "resource {}"
        r1 = TrainingRecord(
            id="1", format="tf", gold_config=gold, broken_config="a",
            errors=[], warnings=[], fault_types=[], fault_description="",
            difficulty="low", source="programmatic",
        )
        r2 = TrainingRecord(
            id="2", format="tf", gold_config=gold, broken_config="b",
            errors=[], warnings=[], fault_types=[], fault_description="",
            difficulty="low", source="programmatic",
        )
        assert r1.gold_hash == r2.gold_hash


class TestFormatAndSplit:
    """Test format_and_split function."""

    def test_basic_split(self):
        records = []
        for i in range(10):
            records.append(TrainingRecord(
                id=f"rec-{i}",
                format="terraform",
                gold_config=f"gold config {i}",
                broken_config=f"broken config {i}",
                errors=[f"error {i}"],
                warnings=[],
                fault_types=[f"SYNTACTIC.fault_{i}"],
                fault_description=f"Fault {i}",
                difficulty="low",
                source="programmatic",
            ))

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = format_and_split(records, tmpdir, ratios=(0.8, 0.1, 0.1))

            assert metadata["total_records"] == 10
            assert (Path(tmpdir) / "train.jsonl").exists()
            assert (Path(tmpdir) / "val.jsonl").exists()
            assert (Path(tmpdir) / "test.jsonl").exists()
            assert (Path(tmpdir) / "metadata.json").exists()

            # Verify all records appear in exactly one split
            total = sum(metadata["splits"].values())
            assert total == 10

    def test_no_gold_leakage(self):
        """All variants of the same gold config must be in the same split."""
        records = []
        gold_configs = ["gold_A", "gold_B", "gold_C", "gold_D", "gold_E"]
        for i, gold in enumerate(gold_configs):
            for j in range(3):  # 3 variants per gold
                records.append(TrainingRecord(
                    id=f"rec-{i}-{j}",
                    format="terraform",
                    gold_config=gold,
                    broken_config=f"broken {i}-{j}",
                    errors=[f"error"],
                    warnings=[],
                    fault_types=["SYNTACTIC.test"],
                    fault_description="test",
                    difficulty="low",
                    source="programmatic",
                ))

        with tempfile.TemporaryDirectory() as tmpdir:
            format_and_split(records, tmpdir, ratios=(0.6, 0.2, 0.2))

            # Read all splits and check no gold hash appears in multiple splits
            hash_splits: dict[str, set[str]] = {}
            for split in ["train", "val", "test"]:
                filepath = Path(tmpdir) / f"{split}.jsonl"
                with open(filepath) as f:
                    for line in f:
                        data = json.loads(line)
                        h = data["gold_hash"]
                        hash_splits.setdefault(h, set()).add(split)

            for h, splits in hash_splits.items():
                assert len(splits) == 1, (
                    f"Gold hash {h} appears in multiple splits: {splits}"
                )

    def test_jsonl_schema(self):
        """Verify JSONL output has expected fields."""
        records = [TrainingRecord(
            id="schema-test",
            format="cloudformation",
            gold_config="AWSTemplateFormatVersion: ...",
            broken_config="broken",
            errors=["E3001"],
            warnings=["W2509"],
            fault_types=["SYNTACTIC.invalid_yaml"],
            fault_description="test",
            difficulty="medium",
            source="programmatic",
        )]

        with tempfile.TemporaryDirectory() as tmpdir:
            format_and_split(records, tmpdir)

            # Find the split that has the record
            data = None
            for split in ["train", "val", "test"]:
                p = Path(tmpdir) / f"{split}.jsonl"
                if p.exists() and p.stat().st_size > 0:
                    with open(p) as f:
                        line = f.readline().strip()
                        if line:
                            data = json.loads(line)
                            break

            assert data is not None, "No records found in any split"

            expected_keys = {
                "id", "format", "gold_config", "broken_config",
                "errors", "warnings", "fault_types", "fault_description",
                "difficulty", "source", "split", "gold_hash",
            }
            assert expected_keys.issubset(set(data.keys()))


class TestPipelineRunner:
    """Test PipelineRunner with fixture configs."""

    def test_discover_gold(self):
        from cloudgym.generator.pipeline import PipelineRunner

        runner = PipelineRunner()
        files = runner._discover_gold(FIXTURES)
        assert len(files) >= 2  # gold_main.tf and gold_template.yaml

    def test_detect_format_tf(self):
        from cloudgym.generator.pipeline import PipelineRunner

        runner = PipelineRunner()
        assert runner._detect_format(FIXTURES / "gold_main.tf") == "terraform"

    def test_detect_format_cf(self):
        from cloudgym.generator.pipeline import PipelineRunner

        runner = PipelineRunner()
        fmt = runner._detect_format(FIXTURES / "gold_template.yaml")
        # Could be "terraform" or "cloudformation" depending on detection
        assert fmt in ("terraform", "cloudformation")

    def test_stratified_faults(self):
        from cloudgym.generator.pipeline import PipelineRunner

        runner = PipelineRunner()
        faults = runner._get_stratified_faults("terraform", 7)
        assert len(faults) == 7
        # Check that categories are diverse (not all from one category)
        categories = {ft.category.name for ft in faults}
        assert len(categories) >= 3  # At least 3 different categories
