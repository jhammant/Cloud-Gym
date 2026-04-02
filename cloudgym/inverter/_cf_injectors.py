"""CloudFormation fault injector functions — one per fault type.

Each injector takes (text, parsed_dict) and returns (modified_text, FaultInjection) or None.
CF faults use dict mutation via yaml/json (fully round-trippable), except for
syntactic faults which do raw text corruption.
"""

from __future__ import annotations

import copy
import json
import random
import re
from typing import Any

from cloudgym.inverter._yaml_cf import cf_dump, cf_load

from cloudgym.inverter._cf_utils import (
    REQUIRED_PROPERTIES,
    RESOURCE_TYPE_TYPOS,
    find_getatt,
    find_ifs,
    find_joins,
    find_refs,
    find_selects,
    find_subs,
    get_condition_names,
    get_parameter_names,
    get_resource_logical_ids,
    get_resource_type,
)
from cloudgym.taxonomy.base import FaultInjection

# Registry mapping fault IDs to injector functions
CF_INJECTOR_REGISTRY: dict[str, Any] = {}


def register_cf_injector(fault_id: str):
    """Decorator to register a CF injector function."""
    def decorator(fn):
        CF_INJECTOR_REGISTRY[fault_id] = fn
        return fn
    return decorator


def _is_json(text: str) -> bool:
    """Check if the template is JSON format."""
    stripped = text.strip()
    return stripped.startswith('{')


def _dump(template: dict, is_json: bool) -> str:
    """Serialize template back to text."""
    if is_json:
        return json.dumps(template, indent=2)
    return cf_dump(template)


# ---------------------------------------------------------------------------
# SYNTACTIC injectors (raw text manipulation)
# ---------------------------------------------------------------------------

@register_cf_injector("SYNTACTIC.invalid_yaml")
def inject_invalid_yaml(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Introduce invalid YAML syntax."""
    if _is_json(text):
        return None

    lines = text.split('\n')
    # Find a non-empty, non-comment line and insert a tab character
    candidates = [
        i for i, line in enumerate(lines)
        if line.strip() and not line.strip().startswith('#') and ':' in line
    ]
    if not candidates:
        return None

    target_line = random.choice(candidates)
    original_line = lines[target_line]
    # Insert a tab at the beginning (YAML forbids tabs for indentation)
    lines[target_line] = '\t' + original_line.lstrip()
    modified = '\n'.join(lines)

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=original_line.strip(),
        modified_snippet=lines[target_line].strip(),
        location=f"line {target_line + 1}",
        description="Inserted tab character in YAML indentation",
    )


@register_cf_injector("SYNTACTIC.invalid_json_template")
def inject_invalid_json_template(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """For JSON templates: add trailing comma."""
    if not _is_json(text):
        return None

    # Find a line with }, or ] and add a trailing comma before it
    lines = text.split('\n')
    candidates = [
        i for i, line in enumerate(lines)
        if line.rstrip().endswith('}') or line.rstrip().endswith('"')
    ]
    if not candidates:
        return None

    target_line = random.choice(candidates)
    original_line = lines[target_line]
    # Add a trailing comma after the value
    lines[target_line] = original_line.rstrip() + ','
    # Also insert after a closing brace line that's followed by another closing brace
    modified = '\n'.join(lines)

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=original_line.strip(),
        modified_snippet=lines[target_line].strip(),
        location=f"line {target_line + 1}",
        description="Added trailing comma to create invalid JSON",
    )


@register_cf_injector("SYNTACTIC.wrong_property_type")
def inject_wrong_property_type(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Set a property to the wrong type (e.g., string where list expected)."""
    resources = parsed.get("Resources", {})
    if not resources:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    for logical_id, resource in template.get("Resources", {}).items():
        props = resource.get("Properties", {})
        for prop_name, prop_value in props.items():
            if isinstance(prop_value, list):
                # Change list to string
                original = prop_value
                template["Resources"][logical_id]["Properties"][prop_name] = "not-a-list"
                modified = _dump(template, is_json)
                return modified, FaultInjection(
                    fault_type=None,
                    original_snippet=f"{prop_name}: {str(original)[:60]}",
                    modified_snippet=f'{prop_name}: "not-a-list"',
                    location=f"Resources.{logical_id}.Properties.{prop_name}",
                    description=f"Changed list property {prop_name} to string",
                )
            if isinstance(prop_value, dict) and not any(
                k.startswith("Fn::") or k in ("Ref",) for k in prop_value
            ):
                # Change dict to string
                original = prop_value
                template["Resources"][logical_id]["Properties"][prop_name] = "not-a-dict"
                modified = _dump(template, is_json)
                return modified, FaultInjection(
                    fault_type=None,
                    original_snippet=f"{prop_name}: {str(original)[:60]}",
                    modified_snippet=f'{prop_name}: "not-a-dict"',
                    location=f"Resources.{logical_id}.Properties.{prop_name}",
                    description=f"Changed dict property {prop_name} to string",
                )

    return None


@register_cf_injector("SYNTACTIC.missing_required_property")
def inject_missing_required_property(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove a required property from a resource."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    for logical_id, resource in resources.items():
        res_type = resource.get("Type", "")
        required = REQUIRED_PROPERTIES.get(res_type, [])
        props = resource.get("Properties", {})

        for req_prop in required:
            if req_prop in props:
                original_val = props[req_prop]
                del template["Resources"][logical_id]["Properties"][req_prop]
                modified = _dump(template, is_json)
                return modified, FaultInjection(
                    fault_type=None,
                    original_snippet=f"{req_prop}: {str(original_val)[:60]}",
                    modified_snippet="(property removed)",
                    location=f"Resources.{logical_id}.Properties.{req_prop}",
                    description=f"Removed required property '{req_prop}' from {res_type}",
                )

    # Fallback: remove any property
    for logical_id, resource in resources.items():
        props = resource.get("Properties", {})
        if props:
            prop_name = next(iter(props))
            original_val = props[prop_name]
            del template["Resources"][logical_id]["Properties"][prop_name]
            modified = _dump(template, is_json)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=f"{prop_name}: {str(original_val)[:60]}",
                modified_snippet="(property removed)",
                location=f"Resources.{logical_id}.Properties.{prop_name}",
                description=f"Removed property '{prop_name}' from {resource.get('Type', 'unknown')}",
            )

    return None


# ---------------------------------------------------------------------------
# REFERENCE injectors
# ---------------------------------------------------------------------------

@register_cf_injector("REFERENCE.broken_ref")
def inject_broken_ref(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Corrupt a !Ref target to point to non-existent resource."""
    refs = find_refs(parsed)
    if not refs:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    # Pick a Ref that points to a resource or parameter
    resource_ids = set(get_resource_logical_ids(parsed))
    param_names = set(get_parameter_names(parsed))
    valid_refs = [
        (target, path) for target, path in refs
        if target in resource_ids or target in param_names
    ]
    if not valid_refs:
        return None

    target, path = random.choice(valid_refs)
    corrupted = target + "Nonexistent"

    # Navigate to the Ref and replace
    obj = template
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]
    if isinstance(obj, dict) and path[-1] in obj:
        obj[path[-1]] = corrupted

    modified = _dump(template, is_json)
    return modified, FaultInjection(
        fault_type=None,
        original_snippet=f"Ref: {target}",
        modified_snippet=f"Ref: {corrupted}",
        location=".".join(path),
        description=f"Changed Ref target from {target} to {corrupted}",
    )


@register_cf_injector("REFERENCE.bad_getatt")
def inject_bad_getatt(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Corrupt a !GetAtt target."""
    getatt_refs = find_getatt(parsed)
    if not getatt_refs:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    value, path = random.choice(getatt_refs)

    # Navigate and corrupt
    obj = template
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]

    if isinstance(obj, dict) and path[-1] in obj:
        original = obj[path[-1]]
        if isinstance(original, list) and len(original) >= 2:
            obj[path[-1]] = [original[0] + "Broken", original[1]]
            corrupted_str = f"{original[0]}Broken.{original[1]}"
        elif isinstance(original, str) and '.' in original:
            parts = original.split('.', 1)
            obj[path[-1]] = parts[0] + "Broken." + parts[1]
            corrupted_str = obj[path[-1]]
        else:
            return None

        modified = _dump(template, is_json)
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=f"GetAtt: {original}",
            modified_snippet=f"GetAtt: {corrupted_str}",
            location=".".join(path),
            description=f"Corrupted GetAtt resource reference",
        )

    return None


@register_cf_injector("REFERENCE.undefined_parameter")
def inject_undefined_parameter(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Delete a parameter from Parameters that is referenced in Resources."""
    params = get_parameter_names(parsed)
    if not params:
        return None

    # Find a parameter that is referenced
    refs = find_refs(parsed)
    referenced_params = [t for t, _ in refs if t in params]
    if not referenced_params:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    param_to_remove = random.choice(referenced_params)
    del template["Parameters"][param_to_remove]

    # Remove empty Parameters section
    if not template["Parameters"]:
        del template["Parameters"]

    modified = _dump(template, is_json)
    return modified, FaultInjection(
        fault_type=None,
        original_snippet=f"Parameter: {param_to_remove}",
        modified_snippet="(parameter removed)",
        location=f"Parameters.{param_to_remove}",
        description=f"Removed parameter '{param_to_remove}' that is referenced in Resources",
    )


# ---------------------------------------------------------------------------
# SEMANTIC injectors
# ---------------------------------------------------------------------------

@register_cf_injector("SEMANTIC.cf_invalid_resource_type")
def inject_cf_invalid_resource_type(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Use a non-existent resource type."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    for logical_id, resource in resources.items():
        res_type = resource.get("Type", "")
        if res_type in RESOURCE_TYPE_TYPOS:
            new_type = RESOURCE_TYPE_TYPOS[res_type]
        elif res_type.startswith("AWS::"):
            # Generic corruption
            parts = res_type.split("::")
            if len(parts) == 3:
                new_type = f"{parts[0]}::{parts[1]}::{parts[2]}Invalid"
            else:
                new_type = res_type + "Invalid"
        else:
            continue

        template["Resources"][logical_id]["Type"] = new_type
        modified = _dump(template, is_json)
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=f"Type: {res_type}",
            modified_snippet=f"Type: {new_type}",
            location=f"Resources.{logical_id}.Type",
            description=f"Changed resource type from {res_type} to {new_type}",
        )

    return None


@register_cf_injector("SEMANTIC.wrong_property_value")
def inject_wrong_property_value(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Set a property to an invalid value."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    # Look for InstanceType, Engine, or other constrained properties
    value_corruptions = {
        "InstanceType": "x1.superlarge",
        "Engine": "postgres-invalid-engine",
        "Runtime": "python2.5",
        "Protocol": "INVALID",
    }

    for logical_id, resource in resources.items():
        props = resource.get("Properties", {})
        for prop_name, bad_val in value_corruptions.items():
            if prop_name in props and isinstance(props[prop_name], str):
                original = props[prop_name]
                template["Resources"][logical_id]["Properties"][prop_name] = bad_val
                modified = _dump(template, is_json)
                return modified, FaultInjection(
                    fault_type=None,
                    original_snippet=f"{prop_name}: {original}",
                    modified_snippet=f"{prop_name}: {bad_val}",
                    location=f"Resources.{logical_id}.Properties.{prop_name}",
                    description=f"Changed {prop_name} to invalid value '{bad_val}'",
                )

    return None


@register_cf_injector("SEMANTIC.cf_bad_ami")
def inject_cf_bad_ami(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Corrupt an AMI ID in ImageId property."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    for logical_id, resource in resources.items():
        props = resource.get("Properties", {})
        if "ImageId" in props and isinstance(props["ImageId"], str):
            original = props["ImageId"]
            template["Resources"][logical_id]["Properties"]["ImageId"] = "INVALID-ami-format"
            modified = _dump(template, is_json)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=f"ImageId: {original}",
                modified_snippet="ImageId: INVALID-ami-format",
                location=f"Resources.{logical_id}.Properties.ImageId",
                description=f"Corrupted AMI ID from {original} to INVALID-ami-format",
            )

    return None


# ---------------------------------------------------------------------------
# DEPENDENCY injectors
# ---------------------------------------------------------------------------

@register_cf_injector("DEPENDENCY.circular_depends_on")
def inject_circular_depends_on(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Create a circular DependsOn chain."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})
    logical_ids = list(resources.keys())

    if len(logical_ids) < 2:
        return None

    id1, id2 = logical_ids[0], logical_ids[1]
    template["Resources"][id1]["DependsOn"] = [id2]
    template["Resources"][id2]["DependsOn"] = [id1]

    modified = _dump(template, is_json)
    return modified, FaultInjection(
        fault_type=None,
        original_snippet="(no circular dependency)",
        modified_snippet=f"{id1} -> {id2} -> {id1}",
        location=f"Resources.{id1} <-> Resources.{id2}",
        description=f"Created circular DependsOn between {id1} and {id2}",
    )


@register_cf_injector("DEPENDENCY.missing_dependency")
def inject_missing_dependency(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove DependsOn from a resource."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    for logical_id, resource in resources.items():
        if "DependsOn" in resource:
            original = resource["DependsOn"]
            del template["Resources"][logical_id]["DependsOn"]
            modified = _dump(template, is_json)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=f"DependsOn: {original}",
                modified_snippet="(DependsOn removed)",
                location=f"Resources.{logical_id}.DependsOn",
                description=f"Removed DependsOn from {logical_id}",
            )

    return None


# ---------------------------------------------------------------------------
# INTRINSIC function injectors
# ---------------------------------------------------------------------------

@register_cf_injector("INTRINSIC.malformed_sub")
def inject_malformed_sub(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Break !Sub syntax (unclosed ${, reference non-existent variable)."""
    subs = find_subs(parsed)
    if not subs:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    value, path = random.choice(subs)

    obj = template
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]

    if isinstance(obj, dict) and path[-1] in obj:
        original = obj[path[-1]]
        if isinstance(original, str):
            # Add an unclosed ${
            obj[path[-1]] = original + " ${UndefinedVar"
        elif isinstance(original, list) and len(original) >= 1 and isinstance(original[0], str):
            obj[path[-1]][0] = original[0] + " ${UndefinedVar"
        else:
            return None

        modified = _dump(template, is_json)
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=f"Sub: {str(original)[:60]}",
            modified_snippet=f"Sub: {str(obj[path[-1]])[:60]}",
            location=".".join(path),
            description="Added unclosed variable reference in !Sub",
        )

    return None


@register_cf_injector("INTRINSIC.wrong_select_index")
def inject_wrong_select_index(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Use !Select with an out-of-bounds index."""
    selects = find_selects(parsed)
    if not selects:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    value, path = random.choice(selects)

    obj = template
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]

    if isinstance(obj, dict) and path[-1] in obj:
        original = obj[path[-1]]
        if isinstance(original, list) and len(original) == 2:
            # Set index to very large number
            obj[path[-1]] = [999, original[1]]
            modified = _dump(template, is_json)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=f"Select: [{original[0]}, ...]",
                modified_snippet="Select: [999, ...]",
                location=".".join(path),
                description="Set !Select index to 999 (out of bounds)",
            )

    return None


@register_cf_injector("INTRINSIC.bad_if_condition")
def inject_bad_if_condition(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Use !If with a condition name not defined in Conditions."""
    ifs = find_ifs(parsed)
    if not ifs:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    value, path = random.choice(ifs)

    obj = template
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]

    if isinstance(obj, dict) and path[-1] in obj:
        original = obj[path[-1]]
        if isinstance(original, list) and len(original) >= 1:
            old_condition = original[0]
            obj[path[-1]][0] = "NonExistentCondition"
            modified = _dump(template, is_json)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=f"If: [{old_condition}, ...]",
                modified_snippet="If: [NonExistentCondition, ...]",
                location=".".join(path),
                description=f"Changed !If condition from '{old_condition}' to 'NonExistentCondition'",
            )

    return None


@register_cf_injector("INTRINSIC.invalid_join")
def inject_invalid_join(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Break !Join syntax (wrong number of arguments)."""
    joins = find_joins(parsed)
    if not joins:
        return None

    is_json = _is_json(text)
    template = copy.deepcopy(parsed)

    value, path = random.choice(joins)

    obj = template
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]

    if isinstance(obj, dict) and path[-1] in obj:
        original = obj[path[-1]]
        # Replace with invalid structure (string instead of [delimiter, list])
        obj[path[-1]] = "invalid-join-not-a-list"
        modified = _dump(template, is_json)
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=f"Join: {str(original)[:60]}",
            modified_snippet='Join: "invalid-join-not-a-list"',
            location=".".join(path),
            description="Changed !Join value to invalid string (should be [delimiter, list])",
        )

    return None


# ---------------------------------------------------------------------------
# SECURITY injectors
# ---------------------------------------------------------------------------

@register_cf_injector("SECURITY.open_ingress")
def inject_open_ingress(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Open security group ingress to 0.0.0.0/0."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    for logical_id, resource in resources.items():
        if resource.get("Type") == "AWS::EC2::SecurityGroup":
            props = resource.get("Properties", {})
            open_rule = {
                "IpProtocol": "tcp",
                "FromPort": 0,
                "ToPort": 65535,
                "CidrIp": "0.0.0.0/0",
            }

            ingress = props.get("SecurityGroupIngress", [])
            if isinstance(ingress, list):
                ingress.append(open_rule)
            else:
                ingress = [open_rule]
            template["Resources"][logical_id]["Properties"]["SecurityGroupIngress"] = ingress

            modified = _dump(template, is_json)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet="(restricted ingress)",
                modified_snippet="CidrIp: 0.0.0.0/0, Ports: 0-65535",
                location=f"Resources.{logical_id}.Properties.SecurityGroupIngress",
                description=f"Added overly permissive ingress rule to {logical_id}",
            )

    return None


@register_cf_injector("SECURITY.cf_missing_encryption")
def inject_cf_missing_encryption(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove encryption configuration from S3 bucket or RDS."""
    is_json = _is_json(text)
    template = copy.deepcopy(parsed)
    resources = template.get("Resources", {})

    encryption_props = {
        "AWS::S3::Bucket": ["BucketEncryption"],
        "AWS::RDS::DBInstance": ["StorageEncrypted", "KmsKeyId"],
        "AWS::EBS::Volume": ["Encrypted", "KmsKeyId"],
        "AWS::DynamoDB::Table": ["SSESpecification"],
    }

    for logical_id, resource in resources.items():
        res_type = resource.get("Type", "")
        if res_type in encryption_props:
            props = resource.get("Properties", {})
            for enc_prop in encryption_props[res_type]:
                if enc_prop in props:
                    original = props[enc_prop]
                    del template["Resources"][logical_id]["Properties"][enc_prop]
                    modified = _dump(template, is_json)
                    return modified, FaultInjection(
                        fault_type=None,
                        original_snippet=f"{enc_prop}: {str(original)[:60]}",
                        modified_snippet="(property removed)",
                        location=f"Resources.{logical_id}.Properties.{enc_prop}",
                        description=f"Removed encryption property '{enc_prop}' from {res_type}",
                    )

    return None
