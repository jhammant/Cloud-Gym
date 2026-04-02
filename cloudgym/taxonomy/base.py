"""Fault taxonomy base definitions for IaC environment inversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class FaultCategory(Enum):
    """High-level categories of IaC faults."""

    SYNTACTIC = auto()
    REFERENCE = auto()
    SEMANTIC = auto()
    DEPENDENCY = auto()
    PROVIDER = auto()
    SECURITY = auto()
    CROSS_RESOURCE = auto()
    INTRINSIC = auto()  # CloudFormation-specific (intrinsic function faults)


class IaCFormat(Enum):
    """Supported Infrastructure-as-Code formats."""

    TERRAFORM = "terraform"
    CLOUDFORMATION = "cloudformation"
    OPENTOFU = "opentofu"


class Severity(Enum):
    """Fault severity / difficulty level."""

    LOW = "low"  # Obvious syntax errors, easy to spot
    MEDIUM = "medium"  # Requires understanding of resource relationships
    HIGH = "high"  # Subtle semantic or cross-resource issues


@dataclass(frozen=True)
class FaultType:
    """A specific type of fault that can be injected into IaC configs.

    Each FaultType defines a category of breakage (e.g., "missing closing brace")
    and carries metadata for taxonomy analysis and benchmark stratification.
    """

    name: str
    category: FaultCategory
    description: str
    severity: Severity
    applicable_formats: frozenset[IaCFormat]
    example_error: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def id(self) -> str:
        """Short identifier: category.name (e.g., SYNTACTIC.missing_brace)."""
        return f"{self.category.name}.{self.name}"


@dataclass
class FaultInjection:
    """Record of a single fault injection applied to a config."""

    fault_type: FaultType
    original_snippet: str
    modified_snippet: str
    location: str  # file path or line range description
    description: str  # human-readable explanation of what was changed


@dataclass
class FaultRegistry:
    """Central registry of all known fault types."""

    _faults: dict[str, FaultType] = field(default_factory=dict)

    def register(self, fault: FaultType) -> FaultType:
        self._faults[fault.id] = fault
        return fault

    def get(self, fault_id: str) -> FaultType | None:
        return self._faults.get(fault_id)

    def list_by_category(self, category: FaultCategory) -> list[FaultType]:
        return [f for f in self._faults.values() if f.category == category]

    def list_by_format(self, fmt: IaCFormat) -> list[FaultType]:
        return [f for f in self._faults.values() if fmt in f.applicable_formats]

    def all(self) -> list[FaultType]:
        return list(self._faults.values())

    def __len__(self) -> int:
        return len(self._faults)


# Global registry instance
REGISTRY = FaultRegistry()
