"""Inversion engine — orchestrates fault injection and validates breaks.

Reads gold config, selects fault type(s), injects faults, validates that
the config is actually broken, and returns structured results.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cloudgym.inverter.programmatic import inject_fault
from cloudgym.taxonomy import REGISTRY
from cloudgym.taxonomy.base import FaultInjection, FaultType
from cloudgym.validator.terraform import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class InversionResult:
    """Result of a single fault injection attempt."""

    gold_config: str
    broken_config: str
    fault_type: FaultType
    injection: FaultInjection
    validation_result: ValidationResult
    attempts: int = 1
    gold_path: str = ""
    iac_format: str = ""


def _detect_format(path: Path) -> str:
    """Detect IaC format from file extension."""
    suffix = path.suffix.lower()
    if suffix == ".tf":
        return "terraform"
    if suffix in (".yaml", ".yml", ".json", ".template"):
        return "cloudformation"
    return "terraform"  # default


def _get_applicable_faults(iac_format: str) -> list[FaultType]:
    """Get fault types applicable to the given format."""
    from cloudgym.taxonomy.base import IaCFormat

    fmt_map = {
        "terraform": IaCFormat.TERRAFORM,
        "opentofu": IaCFormat.OPENTOFU,
        "cloudformation": IaCFormat.CLOUDFORMATION,
    }
    iac_fmt = fmt_map.get(iac_format)
    if iac_fmt is None:
        return []
    return REGISTRY.list_by_format(iac_fmt)


async def _validate_broken(
    broken_content: str, iac_format: str
) -> ValidationResult:
    """Write broken config to temp file and validate."""
    suffix = ".tf" if iac_format in ("terraform", "opentofu") else ".yaml"
    tmpdir = Path(tempfile.mkdtemp(prefix="cloudgym_inv_"))
    tmp_file = tmpdir / f"broken{suffix}"
    tmp_file.write_text(broken_content)

    try:
        if iac_format in ("terraform", "opentofu"):
            from cloudgym.validator.terraform import validate
            return await validate(tmp_file)
        else:
            from cloudgym.validator.cloudformation import validate
            return await validate(tmp_file)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


class InversionEngine:
    """Orchestrates fault injection with validation feedback loop."""

    def __init__(
        self,
        max_retries: int = 3,
        concurrency: int = 5,
        skip_validation: bool = False,
    ):
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self.skip_validation = skip_validation

    async def invert(
        self,
        gold_config_path: str | Path,
        fault_types: list[FaultType] | None = None,
        mode: str = "programmatic",
    ) -> InversionResult | None:
        """Inject a fault into a gold config and validate the break.

        Args:
            gold_config_path: Path to the gold (valid) config file.
            fault_types: Specific fault types to try. If None, auto-selects.
            mode: "programmatic" or "agentic".

        Returns:
            InversionResult if successful, None if all attempts fail.
        """
        async with self._semaphore:
            path = Path(gold_config_path)
            iac_format = _detect_format(path)
            config_content = path.read_text()

            if fault_types is None:
                fault_types = _get_applicable_faults(iac_format)

            if not fault_types:
                logger.warning("No applicable faults for %s", iac_format)
                return None

            for attempt in range(self.max_retries):
                fault_type = fault_types[attempt % len(fault_types)]

                if mode == "agentic":
                    result = await self._try_agentic(
                        config_content, fault_type, iac_format
                    )
                else:
                    result = await self._try_programmatic(
                        config_content, fault_type, iac_format
                    )

                if result is None:
                    continue

                broken_content, injection = result

                # In skip_validation mode, trust the injector
                if self.skip_validation:
                    val_result = ValidationResult(
                        valid=False,
                        errors=[fault_type.example_error or f"Injected {fault_type.id}"],
                    )
                    return InversionResult(
                        gold_config=config_content,
                        broken_config=broken_content,
                        fault_type=fault_type,
                        injection=injection,
                        validation_result=val_result,
                        attempts=attempt + 1,
                        gold_path=str(path),
                        iac_format=iac_format,
                    )

                # Validate that the broken config actually fails
                val_result = await _validate_broken(broken_content, iac_format)

                if not val_result.valid or val_result.errors:
                    return InversionResult(
                        gold_config=config_content,
                        broken_config=broken_content,
                        fault_type=fault_type,
                        injection=injection,
                        validation_result=val_result,
                        attempts=attempt + 1,
                        gold_path=str(path),
                        iac_format=iac_format,
                    )

                # For security faults, check if warnings increased
                if fault_type.category.name == "SECURITY":
                    gold_val = await _validate_broken(config_content, iac_format)
                    if len(val_result.warnings) > len(gold_val.warnings):
                        return InversionResult(
                            gold_config=config_content,
                            broken_config=broken_content,
                            fault_type=fault_type,
                            injection=injection,
                            validation_result=val_result,
                            attempts=attempt + 1,
                            gold_path=str(path),
                            iac_format=iac_format,
                        )

                logger.debug(
                    "Attempt %d: fault %s did not break validation for %s",
                    attempt + 1, fault_type.id, path.name,
                )

            return None

    async def _try_programmatic(
        self,
        config_content: str,
        fault_type: FaultType,
        iac_format: str,
    ) -> tuple[str, FaultInjection] | None:
        """Try programmatic fault injection."""
        return await inject_fault(config_content, fault_type, iac_format)

    async def _try_agentic(
        self,
        config_content: str,
        fault_type: FaultType,
        iac_format: str,
    ) -> tuple[str, FaultInjection] | None:
        """Try agentic (LLM) fault injection."""
        from cloudgym.inverter.agentic import inject_fault_agentic

        broken = await inject_fault_agentic(
            config_content, fault_type.category.name, iac_format
        )
        if broken is None:
            return None

        injection = FaultInjection(
            fault_type=fault_type,
            original_snippet=config_content[:80],
            modified_snippet=broken[:80],
            location="agentic (full config)",
            description=f"LLM-injected {fault_type.category.name} fault",
        )
        return broken, injection


# Convenience function matching original stub signature
async def invert(
    gold_config_path: str,
    fault_types: list[str] | None = None,
    mode: str = "programmatic",
) -> dict | None:
    """Orchestrate fault injection on a gold config.

    Returns a dict with original config, broken config, fault types applied,
    and validation errors.
    """
    engine = InversionEngine()

    ft_objects = None
    if fault_types:
        ft_objects = [REGISTRY.get(fid) for fid in fault_types]
        ft_objects = [ft for ft in ft_objects if ft is not None]

    result = await engine.invert(gold_config_path, ft_objects, mode)
    if result is None:
        return None

    return {
        "gold_config": result.gold_config,
        "broken_config": result.broken_config,
        "fault_type": result.fault_type.id,
        "errors": result.validation_result.errors,
        "warnings": result.validation_result.warnings,
        "attempts": result.attempts,
    }
