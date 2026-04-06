"""Detect IaC format and validate configurations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cloudgym.validator.terraform import ValidationResult


class IaCFormat(Enum):
    TERRAFORM = "terraform"
    CLOUDFORMATION = "cloudformation"
    OPENTOFU = "opentofu"


@dataclass
class DetectionResult:
    """Result of detecting the IaC format of a file."""

    format: IaCFormat
    confidence: float
    path: Path


def detect_format(path: Path) -> DetectionResult:
    """Auto-detect whether a file is Terraform, CloudFormation, or OpenTofu."""
    suffix = path.suffix.lower()
    name = path.name.lower()
    content = path.read_text(errors="replace") if path.is_file() else ""

    # Terraform: .tf files
    if suffix == ".tf":
        return DetectionResult(IaCFormat.TERRAFORM, 1.0, path)

    # CloudFormation: .yaml/.yml/.json with AWSTemplateFormatVersion
    if suffix in (".yaml", ".yml", ".json"):
        if "AWSTemplateFormatVersion" in content:
            return DetectionResult(IaCFormat.CLOUDFORMATION, 1.0, path)
        # Check for common CF patterns
        if "Type: AWS::" in content or '"Type": "AWS::' in content:
            return DetectionResult(IaCFormat.CLOUDFORMATION, 0.9, path)

    # HCL files that aren't .tf (e.g., .hcl)
    if suffix == ".hcl":
        return DetectionResult(IaCFormat.TERRAFORM, 0.8, path)

    # Check content patterns as fallback
    if "resource " in content and "{" in content and "provider " in content:
        return DetectionResult(IaCFormat.TERRAFORM, 0.6, path)
    if "AWSTemplateFormatVersion" in content:
        return DetectionResult(IaCFormat.CLOUDFORMATION, 0.9, path)

    # Default to terraform for unknown
    return DetectionResult(IaCFormat.TERRAFORM, 0.3, path)


async def validate_file(path: Path, fmt: IaCFormat | None = None) -> tuple[IaCFormat, ValidationResult]:
    """Validate a file, auto-detecting format if not specified."""
    if fmt is None:
        detection = detect_format(path)
        fmt = detection.format

    if fmt == IaCFormat.CLOUDFORMATION:
        from cloudgym.validator import cloudformation
        result = await cloudformation.validate(path)
    elif fmt == IaCFormat.OPENTOFU:
        from cloudgym.validator import opentofu
        result = await opentofu.validate(path)
    else:
        from cloudgym.validator import terraform
        result = await terraform.validate(path)

    return fmt, result


def validate_file_sync(path: Path, fmt: IaCFormat | None = None) -> tuple[IaCFormat, ValidationResult]:
    """Synchronous wrapper for validate_file."""
    return asyncio.run(validate_file(path, fmt))
