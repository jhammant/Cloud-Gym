"""CloudFormation validation via cfn-lint."""

from __future__ import annotations

from pathlib import Path

from .terraform import ValidationResult


async def validate(config_path: Path) -> ValidationResult:
    """Validate a CloudFormation template using cfn-lint.

    Uses the cfn-lint Python API directly instead of shelling out.
    """
    try:
        from cfnlint import decode, runner
        from cfnlint.config import ConfigMixIn
    except ImportError:
        return ValidationResult(
            valid=False,
            errors=["cfn-lint is not installed — run `pip install cfn-lint`"],
        )

    config_str = str(config_path)

    try:
        args = ConfigMixIn(["--template", config_str, "--format", "json"])
        lint_runner = runner.Runner(args)
        matches = list(lint_runner.run())
    except Exception as exc:
        return ValidationResult(
            valid=False,
            errors=[f"cfn-lint failed: {exc}"],
            raw_output=str(exc),
        )

    errors = []
    warnings = []

    for match in matches:
        rule_id = match.rule.id if hasattr(match, "rule") else ""
        message = str(match)

        # E = error, W = warning, I = info
        if rule_id.startswith("E"):
            errors.append(message)
        else:
            warnings.append(message)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        raw_output="\n".join(str(m) for m in matches),
    )
