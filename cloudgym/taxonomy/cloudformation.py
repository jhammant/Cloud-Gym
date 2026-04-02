"""CloudFormation fault type definitions."""

from __future__ import annotations

from .base import (
    REGISTRY,
    FaultCategory,
    FaultType,
    IaCFormat,
    Severity,
)

_CF = frozenset({IaCFormat.CLOUDFORMATION})

# ---------------------------------------------------------------------------
# SYNTACTIC faults
# ---------------------------------------------------------------------------

CF_INVALID_YAML = REGISTRY.register(
    FaultType(
        name="invalid_yaml",
        category=FaultCategory.SYNTACTIC,
        description="Introduce invalid YAML syntax (bad indentation, missing colon, tab characters)",
        severity=Severity.LOW,
        applicable_formats=_CF,
        example_error="E0000 Template is not valid YAML",
        tags=frozenset({"yaml", "parse-error"}),
    )
)

CF_WRONG_PROPERTY_TYPE = REGISTRY.register(
    FaultType(
        name="wrong_property_type",
        category=FaultCategory.SYNTACTIC,
        description="Set a property to the wrong type (e.g., string where list expected)",
        severity=Severity.LOW,
        applicable_formats=_CF,
        example_error="E3012 Property value is not of type",
        tags=frozenset({"type-mismatch"}),
    )
)

CF_MISSING_REQUIRED_PROPERTY = REGISTRY.register(
    FaultType(
        name="missing_required_property",
        category=FaultCategory.SYNTACTIC,
        description="Remove a required property from a resource (e.g., ImageId from AWS::EC2::Instance)",
        severity=Severity.LOW,
        applicable_formats=_CF,
        example_error="E3003 Property required but missing",
        tags=frozenset({"required-field"}),
    )
)

CF_INVALID_JSON_TEMPLATE = REGISTRY.register(
    FaultType(
        name="invalid_json_template",
        category=FaultCategory.SYNTACTIC,
        description="For JSON templates: trailing comma, missing quote, unescaped character",
        severity=Severity.LOW,
        applicable_formats=_CF,
        example_error="E0000 Template is not valid JSON",
        tags=frozenset({"json", "parse-error"}),
    )
)

# ---------------------------------------------------------------------------
# REFERENCE faults
# ---------------------------------------------------------------------------

CF_BROKEN_REF = REGISTRY.register(
    FaultType(
        name="broken_ref",
        category=FaultCategory.REFERENCE,
        description="Use !Ref to a logical ID that doesn't exist in the template",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E3012 Ref to resource that does not exist",
        tags=frozenset({"ref", "reference"}),
    )
)

CF_BAD_GETATT = REGISTRY.register(
    FaultType(
        name="bad_getatt",
        category=FaultCategory.REFERENCE,
        description="Use !GetAtt with wrong resource name or non-existent attribute",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E3012 GetAtt attribute does not exist",
        tags=frozenset({"getatt", "reference"}),
    )
)

CF_UNDEFINED_PARAMETER = REGISTRY.register(
    FaultType(
        name="undefined_parameter",
        category=FaultCategory.REFERENCE,
        description="Reference a parameter in Resources that isn't defined in Parameters section",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E3012 Ref to parameter that does not exist",
        tags=frozenset({"parameter", "reference"}),
    )
)

# ---------------------------------------------------------------------------
# SEMANTIC faults
# ---------------------------------------------------------------------------

CF_INVALID_RESOURCE_TYPE = REGISTRY.register(
    FaultType(
        name="cf_invalid_resource_type",
        category=FaultCategory.SEMANTIC,
        description="Use a non-existent resource type (e.g., AWS::EC2::VirtualMachine instead of AWS::EC2::Instance)",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E3001 Invalid or unsupported Type",
        tags=frozenset({"resource-type"}),
    )
)

CF_WRONG_PROPERTY_VALUE = REGISTRY.register(
    FaultType(
        name="wrong_property_value",
        category=FaultCategory.SEMANTIC,
        description="Set a property to an invalid value (e.g., InstanceType: 'x1.super' which doesn't exist)",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E3030 Invalid value for AllowedValues",
        tags=frozenset({"property-value"}),
    )
)

CF_BAD_AMI = REGISTRY.register(
    FaultType(
        name="cf_bad_ami",
        category=FaultCategory.SEMANTIC,
        description="Use an invalid AMI ID format in ImageId property",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E3031 Invalid AMI ID format",
        tags=frozenset({"aws", "ami"}),
    )
)

# ---------------------------------------------------------------------------
# DEPENDENCY faults
# ---------------------------------------------------------------------------

CF_CIRCULAR_DEPENDS_ON = REGISTRY.register(
    FaultType(
        name="circular_depends_on",
        category=FaultCategory.DEPENDENCY,
        description="Create a circular DependsOn chain between resources",
        severity=Severity.HIGH,
        applicable_formats=_CF,
        example_error="E3004 Circular dependency between resources",
        tags=frozenset({"dependency", "cycle"}),
    )
)

CF_MISSING_DEPENDENCY = REGISTRY.register(
    FaultType(
        name="missing_dependency",
        category=FaultCategory.DEPENDENCY,
        description="Remove DependsOn where ordering is critical, causing deploy-time failure",
        severity=Severity.HIGH,
        applicable_formats=_CF,
        example_error="Resource creation failed due to missing dependency",
        tags=frozenset({"dependency"}),
    )
)

# ---------------------------------------------------------------------------
# INTRINSIC function faults (CloudFormation-specific)
# ---------------------------------------------------------------------------

CF_MALFORMED_SUB = REGISTRY.register(
    FaultType(
        name="malformed_sub",
        category=FaultCategory.INTRINSIC,
        description="Break !Sub syntax (e.g., unclosed ${, reference non-existent variable in substitution)",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E1029 Sub variable not found",
        tags=frozenset({"intrinsic", "sub"}),
    )
)

CF_WRONG_SELECT_INDEX = REGISTRY.register(
    FaultType(
        name="wrong_select_index",
        category=FaultCategory.INTRINSIC,
        description="Use !Select with an index that exceeds the list length",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E1028 Select index out of range",
        tags=frozenset({"intrinsic", "select"}),
    )
)

CF_BAD_IF_CONDITION = REGISTRY.register(
    FaultType(
        name="bad_if_condition",
        category=FaultCategory.INTRINSIC,
        description="Use !If with a condition name not defined in the Conditions section",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E8005 Condition not defined",
        tags=frozenset({"intrinsic", "condition"}),
    )
)

CF_INVALID_JOIN = REGISTRY.register(
    FaultType(
        name="invalid_join",
        category=FaultCategory.INTRINSIC,
        description="Break !Join syntax (e.g., wrong number of arguments, non-list second argument)",
        severity=Severity.MEDIUM,
        applicable_formats=_CF,
        example_error="E1022 Invalid Join declaration",
        tags=frozenset({"intrinsic", "join"}),
    )
)

# ---------------------------------------------------------------------------
# SECURITY faults
# ---------------------------------------------------------------------------

CF_OPEN_INGRESS = REGISTRY.register(
    FaultType(
        name="open_ingress",
        category=FaultCategory.SECURITY,
        description="Set security group ingress to 0.0.0.0/0 on all ports",
        severity=Severity.HIGH,
        applicable_formats=_CF,
        example_error="W2509 Security Group ingress open to 0.0.0.0/0",
        tags=frozenset({"aws", "security-group", "networking"}),
    )
)

CF_MISSING_ENCRYPTION = REGISTRY.register(
    FaultType(
        name="cf_missing_encryption",
        category=FaultCategory.SECURITY,
        description="Remove server-side encryption from S3 bucket or RDS storage",
        severity=Severity.HIGH,
        applicable_formats=_CF,
        example_error="W2511 Resource missing encryption configuration",
        tags=frozenset({"encryption", "compliance"}),
    )
)


def get_all_cloudformation_faults() -> list[FaultType]:
    """Return all registered CloudFormation fault types."""
    return REGISTRY.list_by_format(IaCFormat.CLOUDFORMATION)
