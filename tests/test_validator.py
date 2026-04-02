"""Tests for IaC validation wrappers."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from cloudgym.validator.terraform import ValidationResult, validate as tf_validate
from cloudgym.validator.cloudformation import validate as cf_validate


class TestValidationResult:
    def test_valid_result(self):
        r = ValidationResult(valid=True)
        assert r.valid
        assert r.errors == []
        assert r.warnings == []

    def test_invalid_result(self):
        r = ValidationResult(valid=False, errors=["something broke"])
        assert not r.valid
        assert len(r.errors) == 1


class TestTerraformValidator:
    @pytest.mark.asyncio
    async def test_valid_config(self, tmp_path):
        """A minimal valid Terraform config should pass validation."""
        tf_file = tmp_path / "main.tf"
        tf_file.write_text(
            'terraform {\n  required_version = ">= 1.0"\n}\n'
            'variable "name" {\n  type = string\n  default = "test"\n}\n'
            'output "result" {\n  value = var.name\n}\n'
        )
        result = await tf_validate(tmp_path)
        # This test requires terraform CLI — skip if not available
        if "terraform CLI not found" in str(result.errors):
            pytest.skip("terraform CLI not installed")
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    @pytest.mark.asyncio
    async def test_invalid_config(self, tmp_path):
        """A config with broken references should fail validation."""
        tf_file = tmp_path / "main.tf"
        tf_file.write_text(
            'output "broken" {\n  value = var.undefined_var\n}\n'
        )
        result = await tf_validate(tmp_path)
        if "terraform CLI not found" in str(result.errors):
            pytest.skip("terraform CLI not installed")
        assert not result.valid


class TestCloudFormationValidator:
    @pytest.mark.asyncio
    async def test_valid_template(self, tmp_path):
        """A minimal valid CF template should pass validation."""
        cf_file = tmp_path / "template.yaml"
        cf_file.write_text(
            "AWSTemplateFormatVersion: '2010-09-09'\n"
            "Description: Test template\n"
            "Resources:\n"
            "  MyBucket:\n"
            "    Type: AWS::S3::Bucket\n"
        )
        result = await cf_validate(cf_file)
        if "cfn-lint is not installed" in str(result.errors):
            pytest.skip("cfn-lint not installed")
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    @pytest.mark.asyncio
    async def test_invalid_template(self, tmp_path):
        """A template with invalid resource type should fail."""
        cf_file = tmp_path / "bad.yaml"
        cf_file.write_text(
            "AWSTemplateFormatVersion: '2010-09-09'\n"
            "Resources:\n"
            "  MyThing:\n"
            "    Type: AWS::Fake::Resource\n"
            "    Properties:\n"
            "      Name: broken\n"
        )
        result = await cf_validate(cf_file)
        if "cfn-lint is not installed" in str(result.errors):
            pytest.skip("cfn-lint not installed")
        assert not result.valid
