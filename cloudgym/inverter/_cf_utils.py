"""CloudFormation dict manipulation helpers for fault injection."""

from __future__ import annotations

from typing import Any


# Required properties by resource type (non-exhaustive, covers common resources)
REQUIRED_PROPERTIES: dict[str, list[str]] = {
    "AWS::EC2::Instance": ["ImageId"],
    "AWS::EC2::SecurityGroup": ["GroupDescription"],
    "AWS::EC2::Subnet": ["VpcId", "CidrBlock"],
    "AWS::EC2::VPC": ["CidrBlock"],
    "AWS::EC2::InternetGateway": [],
    "AWS::EC2::RouteTable": ["VpcId"],
    "AWS::EC2::Route": ["RouteTableId"],
    "AWS::S3::Bucket": [],
    "AWS::RDS::DBInstance": ["DBInstanceClass", "Engine"],
    "AWS::Lambda::Function": ["Code", "Handler", "Role", "Runtime"],
    "AWS::IAM::Role": ["AssumeRolePolicyDocument"],
    "AWS::IAM::Policy": ["PolicyDocument", "PolicyName"],
    "AWS::SNS::Topic": [],
    "AWS::SQS::Queue": [],
    "AWS::DynamoDB::Table": ["KeySchema", "AttributeDefinitions"],
    "AWS::ECS::Cluster": [],
    "AWS::ECS::TaskDefinition": ["ContainerDefinitions"],
    "AWS::ECS::Service": ["TaskDefinition"],
    "AWS::ElasticLoadBalancingV2::LoadBalancer": [],
    "AWS::ElasticLoadBalancingV2::TargetGroup": [],
    "AWS::AutoScaling::AutoScalingGroup": ["MinSize", "MaxSize"],
    "AWS::AutoScaling::LaunchConfiguration": ["ImageId", "InstanceType"],
    "AWS::CloudWatch::Alarm": [
        "ComparisonOperator", "EvaluationPeriods", "MetricName",
        "Namespace", "Period", "Statistic", "Threshold",
    ],
}

# Common resource type typos for injection
RESOURCE_TYPE_TYPOS: dict[str, str] = {
    "AWS::EC2::Instance": "AWS::EC2::VirtualMachine",
    "AWS::S3::Bucket": "AWS::S3::Storage",
    "AWS::Lambda::Function": "AWS::Lambda::Lambda",
    "AWS::RDS::DBInstance": "AWS::RDS::Database",
    "AWS::IAM::Role": "AWS::IAM::ServiceRole",
    "AWS::EC2::SecurityGroup": "AWS::EC2::FirewallGroup",
    "AWS::EC2::VPC": "AWS::EC2::VirtualPrivateCloud",
    "AWS::DynamoDB::Table": "AWS::DynamoDB::Database",
    "AWS::SNS::Topic": "AWS::SNS::Notification",
    "AWS::SQS::Queue": "AWS::SQS::MessageQueue",
}


def find_refs(template: dict) -> list[tuple[str, list[str]]]:
    """Find all !Ref / Fn::Ref targets in a template.

    Returns list of (ref_target, json_path) pairs.
    """
    results: list[tuple[str, list[str]]] = []
    _walk(template, [], lambda path, k, v: (
        results.append((v, list(path) + [k]))
        if k in ("Ref", "Fn::Ref") and isinstance(v, str)
        else None
    ))
    return results


def find_getatt(template: dict) -> list[tuple[list, list[str]]]:
    """Find all !GetAtt / Fn::GetAtt targets.

    Returns list of (getatt_value, json_path) pairs.
    """
    results: list[tuple[list, list[str]]] = []

    def visitor(path: list, key: str, value: Any) -> None:
        if key in ("GetAtt", "Fn::GetAtt"):
            results.append((value, list(path) + [key]))

    _walk(template, [], visitor)
    return results


def find_subs(template: dict) -> list[tuple[Any, list[str]]]:
    """Find all !Sub / Fn::Sub expressions."""
    results: list[tuple[Any, list[str]]] = []

    def visitor(path: list, key: str, value: Any) -> None:
        if key in ("Sub", "Fn::Sub"):
            results.append((value, list(path) + [key]))

    _walk(template, [], visitor)
    return results


def find_selects(template: dict) -> list[tuple[Any, list[str]]]:
    """Find all !Select / Fn::Select expressions."""
    results: list[tuple[Any, list[str]]] = []

    def visitor(path: list, key: str, value: Any) -> None:
        if key in ("Select", "Fn::Select"):
            results.append((value, list(path) + [key]))

    _walk(template, [], visitor)
    return results


def find_ifs(template: dict) -> list[tuple[Any, list[str]]]:
    """Find all !If / Fn::If expressions."""
    results: list[tuple[Any, list[str]]] = []

    def visitor(path: list, key: str, value: Any) -> None:
        if key in ("If", "Fn::If"):
            results.append((value, list(path) + [key]))

    _walk(template, [], visitor)
    return results


def find_joins(template: dict) -> list[tuple[Any, list[str]]]:
    """Find all !Join / Fn::Join expressions."""
    results: list[tuple[Any, list[str]]] = []

    def visitor(path: list, key: str, value: Any) -> None:
        if key in ("Join", "Fn::Join"):
            results.append((value, list(path) + [key]))

    _walk(template, [], visitor)
    return results


def get_resource_logical_ids(template: dict) -> list[str]:
    """Get all logical IDs from the Resources section."""
    resources = template.get("Resources", {})
    if isinstance(resources, dict):
        return list(resources.keys())
    return []


def get_parameter_names(template: dict) -> list[str]:
    """Get all parameter names from the Parameters section."""
    params = template.get("Parameters", {})
    if isinstance(params, dict):
        return list(params.keys())
    return []


def get_condition_names(template: dict) -> list[str]:
    """Get all condition names from the Conditions section."""
    conditions = template.get("Conditions", {})
    if isinstance(conditions, dict):
        return list(conditions.keys())
    return []


def get_resource_type(template: dict, logical_id: str) -> str | None:
    """Get the Type of a resource by logical ID."""
    resources = template.get("Resources", {})
    resource = resources.get(logical_id, {})
    return resource.get("Type")


def set_nested(d: dict, path: list[str], value: Any) -> None:
    """Set a value at a nested path in a dict."""
    for key in path[:-1]:
        if isinstance(d, dict):
            d = d.setdefault(key, {})
        elif isinstance(d, list) and key.isdigit():
            d = d[int(key)]
        else:
            return
    if isinstance(d, dict) and path:
        d[path[-1]] = value


def get_nested(d: dict, path: list[str]) -> Any:
    """Get a value at a nested path in a dict."""
    for key in path:
        if isinstance(d, dict):
            d = d.get(key)
        elif isinstance(d, list) and key.isdigit():
            idx = int(key)
            d = d[idx] if idx < len(d) else None
        else:
            return None
        if d is None:
            return None
    return d


def walk_template(template: dict, visitor_fn: Any) -> None:
    """Recursively walk a CF template dict, calling visitor_fn(path, key, value)."""
    _walk(template, [], visitor_fn)


def _walk(obj: Any, path: list, visitor: Any) -> None:
    """Internal recursive walker."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            visitor(path, key, value)
            _walk(value, path + [key], visitor)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk(item, path + [str(i)], visitor)
