"""Programmatic (rule-based) fault injection for IaC configs.

Routes fault injection requests to the appropriate TF or CF injector function
based on the fault type and IaC format.
"""

from __future__ import annotations

import json
import logging

from cloudgym.inverter._cf_injectors import CF_INJECTOR_REGISTRY
from cloudgym.inverter._tf_injectors import TF_INJECTOR_REGISTRY
from cloudgym.taxonomy.base import FaultInjection, FaultType

logger = logging.getLogger(__name__)

# Combined registry
INJECTOR_REGISTRY: dict[str, dict] = {
    "terraform": TF_INJECTOR_REGISTRY,
    "opentofu": TF_INJECTOR_REGISTRY,  # Same injectors as TF
    "cloudformation": CF_INJECTOR_REGISTRY,
}


def _parse_config(content: str, iac_format: str) -> dict | None:
    """Parse config content into a dict for structural analysis."""
    if iac_format in ("terraform", "opentofu"):
        try:
            import hcl2
            import io
            return hcl2.load(io.StringIO(content))
        except Exception:
            return {}
    else:
        # CloudFormation — try YAML then JSON
        from cloudgym.inverter._yaml_cf import cf_load
        try:
            return cf_load(content)
        except Exception:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {}


async def inject_fault(
    config_content: str,
    fault_type: FaultType,
    iac_format: str,
) -> tuple[str, FaultInjection] | None:
    """Inject a specific fault into an IaC config.

    Args:
        config_content: The original (gold) config content.
        fault_type: The type of fault to inject.
        iac_format: One of "terraform", "cloudformation", "opentofu".

    Returns:
        Tuple of (broken_config_content, injection_record) or None if fault
        is not applicable to this config.
    """
    registry = INJECTOR_REGISTRY.get(iac_format)
    if registry is None:
        logger.warning("No injector registry for format: %s", iac_format)
        return None

    injector_fn = registry.get(fault_type.id)
    if injector_fn is None:
        logger.debug("No injector for fault %s in format %s", fault_type.id, iac_format)
        return None

    parsed = _parse_config(config_content, iac_format)
    if parsed is None:
        logger.warning("Failed to parse config for format %s", iac_format)
        return None

    try:
        result = injector_fn(config_content, parsed)
    except Exception:
        logger.exception("Injector %s raised an exception", fault_type.id)
        return None

    if result is None:
        return None

    broken_content, injection = result
    injection.fault_type = fault_type

    # Verify the injection actually changed something
    if broken_content == config_content:
        logger.debug("Injector %s produced no change", fault_type.id)
        return None

    return broken_content, injection
