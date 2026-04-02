"""OpenTofu validation via tofu CLI."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from .terraform import ValidationResult


async def validate(config_path: Path) -> ValidationResult:
    """Validate an OpenTofu configuration directory or file.

    Runs `tofu init -backend=false` followed by `tofu validate -json`.
    Mirrors the Terraform validator but uses the `tofu` binary.
    """
    tofu_bin = shutil.which("tofu")
    if tofu_bin is None:
        return ValidationResult(
            valid=False,
            errors=["tofu CLI not found in PATH"],
        )

    if config_path.is_file():
        tmpdir = Path(tempfile.mkdtemp(prefix="cloudgym_tofu_"))
        shutil.copy2(config_path, tmpdir / config_path.name)
        work_dir = tmpdir
        cleanup = True
    else:
        work_dir = config_path
        cleanup = False

    try:
        return await _run_tofu(work_dir)
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


async def _run_tofu(work_dir: Path) -> ValidationResult:
    """Run tofu init + validate in the given directory."""
    init_proc = await asyncio.create_subprocess_exec(
        "tofu",
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
            errors=[f"tofu init failed: {stderr_text}"],
            raw_output=stderr_text,
        )

    val_proc = await asyncio.create_subprocess_exec(
        "tofu",
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
            errors=[f"Failed to parse tofu validate output: {raw}"],
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
