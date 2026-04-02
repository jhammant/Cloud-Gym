"""Fault taxonomy for IaC environment inversion.

Importing this package auto-registers all fault types in the global REGISTRY.
"""

from cloudgym.taxonomy.base import REGISTRY  # noqa: F401

# Import submodules to trigger fault registration
import cloudgym.taxonomy.terraform  # noqa: F401
import cloudgym.taxonomy.cloudformation  # noqa: F401
