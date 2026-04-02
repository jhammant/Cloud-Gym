"""Terraform validation via terraform CLI."""

from __future__ import annotations

import asyncio
import json
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of validating an IaC configuration."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_output: str = ""


async def validate(config_path: Path) -> ValidationResult:
    """Validate a Terraform configuration directory or file.

    Runs `terraform init -backend=false` followed by `terraform validate -json`.
    If config_path is a single .tf file, copies it to a temp directory first.
    """
    terraform_bin = shutil.which("terraform")
    if terraform_bin is None:
        return ValidationResult(
            valid=False,
            errors=["terraform CLI not found in PATH"],
        )

    # If single file, create a temp dir with just that file
    if config_path.is_file():
        tmpdir = Path(tempfile.mkdtemp(prefix="cloudgym_tf_"))
        shutil.copy2(config_path, tmpdir / config_path.name)
        work_dir = tmpdir
        cleanup = True
    else:
        work_dir = config_path
        cleanup = False

    try:
        return await _run_terraform(work_dir)
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


async def _run_terraform(work_dir: Path) -> ValidationResult:
    """Run terraform init + validate in the given directory."""
    # terraform init -backend=false
    init_proc = await asyncio.create_subprocess_exec(
        "terraform",
        "init",
        "-backend=false",
        "-no-color",
        cwd=work_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    init_stdout, init_stderr = await init_proc.communicate()

    if init_proc.returncode != 0:
        stderr_text = init_stderr.decode(errors="replace")
        return ValidationResult(
            valid=False,
            errors=[f"terraform init failed: {stderr_text}"],
            raw_output=stderr_text,
        )

    # terraform validate -json
    val_proc = await asyncio.create_subprocess_exec(
        "terraform",
        "validate",
        "-json",
        "-no-color",
        cwd=work_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    val_stdout, val_stderr = await val_proc.communicate()

    raw = val_stdout.decode(errors="replace")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return ValidationResult(
            valid=False,
            errors=[f"Failed to parse terraform validate output: {raw}"],
            raw_output=raw,
        )

    errors = []
    warnings = []
    for diag in result.get("diagnostics", []):
        msg = diag.get("summary", "")
        detail = diag.get("detail", "")
        full_msg = f"{msg}: {detail}" if detail else msg

        if diag.get("severity") == "error":
            errors.append(full_msg)
        else:
            warnings.append(full_msg)

    return ValidationResult(
        valid=result.get("valid", False),
        errors=errors,
        warnings=warnings,
        raw_output=raw,
    )
