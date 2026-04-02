"""LLM-based (agentic) fault injection via Ollama.

Sends gold config + fault category prompt to local LLM with quality gates:
- Similarity check (difflib)
- Diff size check (< 20% lines changed)
- Validation check (must fail terraform validate / cfn-lint)
"""

from __future__ import annotations

import difflib
import logging

from cloudgym.utils.config import InverterConfig
from cloudgym.utils.ollama import OllamaClient

logger = logging.getLogger(__name__)


async def inject_fault_agentic(
    config_content: str,
    fault_category: str,
    iac_format: str,
    config: InverterConfig | None = None,
) -> str | None:
    """Use a local LLM to inject a realistic fault into an IaC config.

    Quality gates:
    - Similarity >= 0.7 (reject if LLM rewrote everything)
    - < 20% of lines changed
    - Output must be non-empty

    Returns the broken config content, or None if quality gates fail.
    """
    if config is None:
        config = InverterConfig()

    client = OllamaClient(config=config)

    try:
        broken = await client.inject_fault(config_content, fault_category, iac_format)
    except Exception:
        logger.exception("Ollama inject_fault failed")
        return None

    if not broken or not broken.strip():
        logger.debug("LLM returned empty response")
        return None

    # Strip markdown fences if present
    broken = _strip_fences(broken)

    # Quality gate 1: similarity check
    similarity = difflib.SequenceMatcher(None, config_content, broken).ratio()
    if similarity < 0.7:
        logger.debug("LLM output too dissimilar (%.2f < 0.7)", similarity)
        return None

    # Quality gate 2: diff size check (< 20% of lines changed)
    orig_lines = config_content.splitlines()
    broken_lines = broken.splitlines()
    diff = list(difflib.unified_diff(orig_lines, broken_lines, lineterm=""))
    changed_lines = sum(1 for line in diff if line.startswith('+') or line.startswith('-'))
    # Subtract header lines (--- and +++)
    changed_lines = max(0, changed_lines - 2)
    total_lines = max(len(orig_lines), 1)

    if changed_lines / total_lines > 0.2:
        logger.debug(
            "LLM changed too many lines (%d/%d = %.0f%%)",
            changed_lines, total_lines, 100 * changed_lines / total_lines,
        )
        return None

    # Quality gate 3: must actually be different
    if broken.strip() == config_content.strip():
        logger.debug("LLM output identical to input")
        return None

    return broken


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
