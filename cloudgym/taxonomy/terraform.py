"""Terraform / OpenTofu fault type definitions."""

from __future__ import annotations

from .base import (
    REGISTRY,
    FaultCategory,
    FaultType,
    IaCFormat,
    Severity,
)

_TF = frozenset({IaCFormat.TERRAFORM, IaCFormat.OPENTOFU})

# ---------------------------------------------------------------------------
# SYNTACTIC faults
# ---------------------------------------------------------------------------

TF_MISSING_CLOSING_BRACE = REGISTRY.register(
    FaultType(
        name="missing_closing_brace",
        category=FaultCategory.SYNTACTIC,
        description="Remove a closing brace from a resource, variable, or output block",
        severity=Severity.LOW,
        applicable_formats=_TF,
        example_error='Error: Argument or block definition required',
        tags=frozenset({"hcl", "parse-error"}),
    )
)

TF_WRONG_ATTRIBUTE_TYPE = REGISTRY.register(
    FaultType(
        name="wrong_attribute_type",
        category=FaultCategory.SYNTACTIC,
        description="Assign a string where a number/bool is expected or vice versa (e.g., port = \"eighty\")",
        severity=Severity.LOW,
        applicable_formats=_TF,
        example_error="Error: Invalid value for variable",
        tags=frozenset({"type-mismatch"}),
    )
)

TF_INVALID_HCL_SYNTAX = REGISTRY.register(
    FaultType(
        name="invalid_hcl_syntax",
        category=FaultCategory.SYNTACTIC,
        description="Introduce invalid HCL syntax (e.g., missing equals sign, stray comma)",
        severity=Severity.LOW,
        applicable_formats=_TF,
        example_error="Error: Invalid expression",
        tags=frozenset({"hcl", "parse-error"}),
    )
)

TF_MISSING_REQUIRED_ARGUMENT = REGISTRY.register(
    FaultType(
        name="missing_required_argument",
        category=FaultCategory.SYNTACTIC,
        description="Remove a required argument from a resource block (e.g., ami from aws_instance)",
        severity=Severity.LOW,
        applicable_formats=_TF,
        example_error='Error: Missing required argument',
        tags=frozenset({"required-field"}),
    )
)

# ---------------------------------------------------------------------------
# REFERENCE faults
# ---------------------------------------------------------------------------

TF_UNDEFINED_VARIABLE = REGISTRY.register(
    FaultType(
        name="undefined_variable",
        category=FaultCategory.REFERENCE,
        description="Reference a variable that is not declared (var.nonexistent)",
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error='Error: Reference to undeclared input variable',
        tags=frozenset({"variable", "reference"}),
    )
)

TF_BAD_RESOURCE_REFERENCE = REGISTRY.register(
    FaultType(
        name="bad_resource_reference",
        category=FaultCategory.REFERENCE,
        description="Reference a non-existent resource or misspell resource name in an expression",
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error='Error: Reference to undeclared resource',
        tags=frozenset({"resource", "reference"}),
    )
)

TF_BROKEN_MODULE_SOURCE = REGISTRY.register(
    FaultType(
        name="broken_module_source",
        category=FaultCategory.REFERENCE,
        description="Set module source to an invalid path or non-existent registry module",
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error="Error: Failed to download module",
        tags=frozenset({"module", "source"}),
    )
)

# ---------------------------------------------------------------------------
# SEMANTIC faults
# ---------------------------------------------------------------------------

TF_INVALID_RESOURCE_TYPE = REGISTRY.register(
    FaultType(
        name="invalid_resource_type",
        category=FaultCategory.SEMANTIC,
        description='Use a non-existent resource type (e.g., aws_ec2_instance instead of aws_instance)',
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error='Error: Invalid resource type',
        tags=frozenset({"resource-type"}),
    )
)

TF_BAD_AMI_FORMAT = REGISTRY.register(
    FaultType(
        name="bad_ami_format",
        category=FaultCategory.SEMANTIC,
        description="Use an invalid AMI ID format (e.g., missing ami- prefix, wrong length)",
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error="Error: expected AMI ID to match pattern",
        tags=frozenset({"aws", "ami"}),
    )
)

TF_INVALID_CIDR = REGISTRY.register(
    FaultType(
        name="invalid_cidr",
        category=FaultCategory.SEMANTIC,
        description="Use an invalid CIDR block (e.g., 10.0.0.0/33, 999.0.0.0/8)",
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error="Error: invalid CIDR address",
        tags=frozenset({"networking", "cidr"}),
    )
)

TF_INVALID_REGION = REGISTRY.register(
    FaultType(
        name="invalid_region",
        category=FaultCategory.SEMANTIC,
        description="Specify a non-existent AWS/Azure/GCP region",
        severity=Severity.LOW,
        applicable_formats=_TF,
        example_error="Error: Invalid AWS Region",
        tags=frozenset({"provider", "region"}),
    )
)

# ---------------------------------------------------------------------------
# DEPENDENCY faults
# ---------------------------------------------------------------------------

TF_CIRCULAR_DEPENDENCY = REGISTRY.register(
    FaultType(
        name="circular_dependency",
        category=FaultCategory.DEPENDENCY,
        description="Create a circular reference between two resources via depends_on or expressions",
        severity=Severity.HIGH,
        applicable_formats=_TF,
        example_error="Error: Cycle",
        tags=frozenset({"dependency", "cycle"}),
    )
)

TF_MISSING_DEPENDS_ON = REGISTRY.register(
    FaultType(
        name="missing_depends_on",
        category=FaultCategory.DEPENDENCY,
        description="Remove an implicit dependency, causing ordering issues at plan time",
        severity=Severity.HIGH,
        applicable_formats=_TF,
        example_error="Error: Reference to undeclared resource",
        tags=frozenset({"dependency"}),
    )
)

# ---------------------------------------------------------------------------
# PROVIDER faults
# ---------------------------------------------------------------------------

TF_MISSING_PROVIDER = REGISTRY.register(
    FaultType(
        name="missing_provider",
        category=FaultCategory.PROVIDER,
        description="Remove the required_providers block or provider configuration",
        severity=Severity.MEDIUM,
        applicable_formats=_TF,
        example_error="Error: Failed to query available provider packages",
        tags=frozenset({"provider"}),
    )
)

TF_VERSION_CONSTRAINT_MISMATCH = REGISTRY.register(
    FaultType(
        name="version_constraint_mismatch",
        category=FaultCategory.PROVIDER,
        description="Set an impossible provider version constraint (e.g., >= 99.0)",
        severity=Severity.LOW,
        applicable_formats=_TF,
        example_error="Error: Failed to query available provider packages",
        tags=frozenset({"provider", "version"}),
    )
)

# ---------------------------------------------------------------------------
# SECURITY faults
# ---------------------------------------------------------------------------

TF_OVERLY_PERMISSIVE_SG = REGISTRY.register(
    FaultType(
        name="overly_permissive_security_group",
        category=FaultCategory.SECURITY,
        description="Open a security group to 0.0.0.0/0 on all ports",
        severity=Severity.HIGH,
        applicable_formats=_TF,
        example_error="Warning: Security group rule allows ingress from 0.0.0.0/0",
        tags=frozenset({"aws", "security-group", "networking"}),
    )
)

TF_MISSING_ENCRYPTION = REGISTRY.register(
    FaultType(
        name="missing_encryption",
        category=FaultCategory.SECURITY,
        description="Remove encryption configuration from S3 bucket, EBS volume, or RDS instance",
        severity=Severity.HIGH,
        applicable_formats=_TF,
        example_error="Warning: Resource missing encryption configuration",
        tags=frozenset({"encryption", "compliance"}),
    )
)

# ---------------------------------------------------------------------------
# CROSS_RESOURCE faults
# ---------------------------------------------------------------------------

TF_SUBNET_VPC_MISMATCH = REGISTRY.register(
    FaultType(
        name="subnet_vpc_mismatch",
        category=FaultCategory.CROSS_RESOURCE,
        description="Reference a subnet that belongs to a different VPC than expected",
        severity=Severity.HIGH,
        applicable_formats=_TF,
        example_error="Error: InvalidSubnet",
        tags=frozenset({"aws", "networking", "vpc"}),
    )
)

TF_WRONG_AZ_REFERENCE = REGISTRY.register(
    FaultType(
        name="wrong_az_reference",
        category=FaultCategory.CROSS_RESOURCE,
        description="Use an availability zone that doesn't match the subnet or region",
        severity=Severity.HIGH,
        applicable_formats=_TF,
        example_error="Error: InvalidParameterValue: Invalid availability zone",
        tags=frozenset({"aws", "availability-zone"}),
    )
)


def get_all_terraform_faults() -> list[FaultType]:
    """Return all registered Terraform fault types."""
    return REGISTRY.list_by_format(IaCFormat.TERRAFORM)
