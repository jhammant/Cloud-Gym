"""Terraform fault injector functions — one per fault type.

Each injector takes (text, parsed_dict) and returns (modified_text, FaultInjection) or None.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from cloudgym.inverter._hcl_utils import (
    find_all_attributes,
    find_block_boundaries,
    find_resource_blocks,
    find_resource_refs,
    find_variable_refs,
    remove_lines,
    replace_value,
)
from cloudgym.taxonomy.base import FaultInjection

if TYPE_CHECKING:
    pass

# Registry mapping fault IDs to injector functions
TF_INJECTOR_REGISTRY: dict[str, Any] = {}


def register_tf_injector(fault_id: str):
    """Decorator to register a TF injector function."""
    def decorator(fn):
        TF_INJECTOR_REGISTRY[fault_id] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# SYNTACTIC injectors
# ---------------------------------------------------------------------------

@register_tf_injector("SYNTACTIC.missing_closing_brace")
def inject_missing_closing_brace(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove a closing brace from a resource block."""
    blocks = find_resource_blocks(text)
    if not blocks:
        return None

    res_type, res_name, start, end = random.choice(blocks)
    # Find the last closing brace of this block
    block_text = text[start:end]
    last_brace = block_text.rfind('}')
    if last_brace < 0:
        return None

    abs_pos = start + last_brace
    modified = text[:abs_pos] + text[abs_pos + 1:]

    return modified, FaultInjection(
        fault_type=None,  # Will be set by caller
        original_snippet=text[max(0, abs_pos - 20):abs_pos + 20],
        modified_snippet=modified[max(0, abs_pos - 20):abs_pos + 19],
        location=f"resource \"{res_type}\" \"{res_name}\"",
        description=f"Removed closing brace from resource {res_type}.{res_name}",
    )


@register_tf_injector("SYNTACTIC.wrong_attribute_type")
def inject_wrong_attribute_type(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Assign a string where a number/bool is expected."""
    blocks = find_resource_blocks(text)
    if not blocks:
        return None

    res_type, res_name, start, end = random.choice(blocks)
    attrs = find_all_attributes(text, start, end)
    if not attrs:
        return None

    # Look for numeric or bool attributes
    lines = text.split('\n')
    for attr_name, line_num in attrs:
        line = lines[line_num]
        # Match numeric values
        m = re.search(r'=\s*(\d+)', line)
        if m:
            old_val = m.group(1)
            new_val = f'"{attr_name}_value"'
            modified = replace_value(text, line_num, old_val, new_val)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=line.strip(),
                modified_snippet=lines[line_num].replace(old_val, new_val, 1).strip() if line_num < len(lines) else "",
                location=f"resource \"{res_type}\" \"{res_name}\", attribute {attr_name}",
                description=f"Changed numeric value to string for {attr_name}",
            )
        # Match bool values
        m = re.search(r'=\s*(true|false)', line)
        if m:
            old_val = m.group(1)
            new_val = '"yes"'
            modified = replace_value(text, line_num, old_val, new_val)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=line.strip(),
                modified_snippet=lines[line_num].replace(old_val, new_val, 1).strip() if line_num < len(lines) else "",
                location=f"resource \"{res_type}\" \"{res_name}\", attribute {attr_name}",
                description=f"Changed boolean to string for {attr_name}",
            )

    return None


@register_tf_injector("SYNTACTIC.invalid_hcl_syntax")
def inject_invalid_hcl_syntax(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Introduce invalid HCL syntax (remove an equals sign)."""
    blocks = find_resource_blocks(text)
    if not blocks:
        return None

    res_type, res_name, start, end = random.choice(blocks)
    attrs = find_all_attributes(text, start, end)
    if not attrs:
        return None

    attr_name, line_num = random.choice(attrs)
    lines = text.split('\n')
    original_line = lines[line_num]
    # Remove the equals sign
    modified_line = re.sub(r'\s*=\s*', ' ', original_line, count=1)
    lines[line_num] = modified_line
    modified = '\n'.join(lines)

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=original_line.strip(),
        modified_snippet=modified_line.strip(),
        location=f"resource \"{res_type}\" \"{res_name}\", line {line_num + 1}",
        description=f"Removed equals sign from attribute {attr_name}",
    )


@register_tf_injector("SYNTACTIC.missing_required_argument")
def inject_missing_required_argument(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove a required argument line from a resource block."""
    blocks = find_resource_blocks(text)
    if not blocks:
        return None

    # Known required arguments per resource type
    required_args = {
        "aws_instance": ["ami", "instance_type"],
        "aws_launch_configuration": ["image_id", "instance_type"],
        "aws_security_group": ["name"],
        "aws_subnet": ["vpc_id", "cidr_block"],
        "aws_vpc": ["cidr_block"],
        "aws_s3_bucket": [],
        "aws_db_instance": ["engine", "instance_class"],
        "aws_lambda_function": ["function_name", "handler", "runtime", "role"],
        "aws_iam_role": ["assume_role_policy"],
    }

    random.shuffle(blocks)
    for res_type, res_name, start, end in blocks:
        reqs = required_args.get(res_type, [])
        if not reqs:
            continue

        attrs = find_all_attributes(text, start, end)
        for attr_name, line_num in attrs:
            if attr_name in reqs:
                original_line = text.split('\n')[line_num]
                modified = remove_lines(text, line_num, line_num)
                return modified, FaultInjection(
                    fault_type=None,
                    original_snippet=original_line.strip(),
                    modified_snippet="(line removed)",
                    location=f"resource \"{res_type}\" \"{res_name}\"",
                    description=f"Removed required argument '{attr_name}' from {res_type}.{res_name}",
                )

    # Fallback: remove any attribute from any block
    res_type, res_name, start, end = blocks[0]
    attrs = find_all_attributes(text, start, end)
    if attrs:
        attr_name, line_num = attrs[0]
        original_line = text.split('\n')[line_num]
        modified = remove_lines(text, line_num, line_num)
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=original_line.strip(),
            modified_snippet="(line removed)",
            location=f"resource \"{res_type}\" \"{res_name}\"",
            description=f"Removed argument '{attr_name}' from {res_type}.{res_name}",
        )

    return None


# ---------------------------------------------------------------------------
# REFERENCE injectors
# ---------------------------------------------------------------------------

@register_tf_injector("REFERENCE.undefined_variable")
def inject_undefined_variable(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Corrupt a var.X reference to point to a non-existent variable."""
    refs = find_variable_refs(text)
    if not refs:
        return None

    var_name, offset = random.choice(refs)
    corrupted_name = var_name + "_undefined"
    old_ref = f"var.{var_name}"
    new_ref = f"var.{corrupted_name}"
    modified = text[:offset] + new_ref + text[offset + len(old_ref):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_ref,
        modified_snippet=new_ref,
        location=f"offset {offset}",
        description=f"Changed variable reference from {old_ref} to {new_ref}",
    )


@register_tf_injector("REFERENCE.bad_resource_reference")
def inject_bad_resource_reference(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Misspell a resource reference in an expression."""
    refs = find_resource_refs(text)
    if not refs:
        return None

    res_type, res_name, offset = random.choice(refs)
    corrupted_name = res_name + "_typo"
    old_ref = f"{res_type}.{res_name}"
    new_ref = f"{res_type}.{corrupted_name}"
    modified = text[:offset] + new_ref + text[offset + len(old_ref):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_ref,
        modified_snippet=new_ref,
        location=f"offset {offset}",
        description=f"Misspelled resource reference from {old_ref} to {new_ref}",
    )


@register_tf_injector("REFERENCE.broken_module_source")
def inject_broken_module_source(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Break a module source path."""
    module_blocks = find_block_boundaries(text, "module")
    if not module_blocks:
        return None

    start, end = random.choice(module_blocks)
    lines = text.split('\n')
    base_line = text[:start].count('\n')
    block_lines = text[start:end].split('\n')

    for i, line in enumerate(block_lines):
        if re.match(r'\s*source\s*=', line):
            line_num = base_line + i
            original_line = lines[line_num]
            # Replace source value with broken path
            modified_line = re.sub(
                r'(source\s*=\s*)"[^"]*"',
                r'\1"./nonexistent_module_path"',
                original_line,
            )
            lines[line_num] = modified_line
            modified = '\n'.join(lines)
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=original_line.strip(),
                modified_snippet=modified_line.strip(),
                location=f"module block, line {line_num + 1}",
                description="Changed module source to non-existent path",
            )

    return None


# ---------------------------------------------------------------------------
# SEMANTIC injectors
# ---------------------------------------------------------------------------

@register_tf_injector("SEMANTIC.invalid_resource_type")
def inject_invalid_resource_type(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Use a non-existent resource type name."""
    blocks = find_resource_blocks(text)
    if not blocks:
        return None

    res_type, res_name, start, end = random.choice(blocks)
    # Corrupt the resource type
    type_typos = {
        "aws_instance": "aws_ec2_instance",
        "aws_s3_bucket": "aws_s3_storage",
        "aws_lambda_function": "aws_lambda",
        "aws_security_group": "aws_firewall_group",
        "aws_vpc": "aws_virtual_private_cloud",
        "aws_subnet": "aws_network_subnet",
        "aws_db_instance": "aws_rds_database",
        "aws_iam_role": "aws_iam_service_role",
    }

    new_type = type_typos.get(res_type, res_type + "_invalid")
    old_decl = f'resource "{res_type}"'
    new_decl = f'resource "{new_type}"'
    modified = text[:start] + text[start:end].replace(old_decl, new_decl, 1) + text[end:]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_decl,
        modified_snippet=new_decl,
        location=f"resource \"{res_type}\" \"{res_name}\"",
        description=f"Changed resource type from {res_type} to {new_type}",
    )


@register_tf_injector("SEMANTIC.bad_ami_format")
def inject_bad_ami_format(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Corrupt an AMI ID to invalid format."""
    # Find ami = "ami-XXXX" patterns
    m = re.search(r'(ami\s*=\s*")(ami-[0-9a-f]+)(")', text)
    if not m:
        # Also try image_id
        m = re.search(r'(image_id\s*=\s*")(ami-[0-9a-f]+)(")', text)
    if not m:
        return None

    old_ami = m.group(2)
    bad_ami = "INVALID-ami-format"
    modified = text[:m.start(2)] + bad_ami + text[m.end(2):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_ami,
        modified_snippet=bad_ami,
        location=f"AMI ID at offset {m.start(2)}",
        description=f"Corrupted AMI ID from {old_ami} to {bad_ami}",
    )


@register_tf_injector("SEMANTIC.invalid_cidr")
def inject_invalid_cidr(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Replace a valid CIDR with an invalid one."""
    m = re.search(r'"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2})"', text)
    if not m:
        return None

    old_cidr = m.group(1)
    bad_cidr = "999.999.999.999/33"
    modified = text[:m.start(1)] + bad_cidr + text[m.end(1):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_cidr,
        modified_snippet=bad_cidr,
        location=f"CIDR block at offset {m.start(1)}",
        description=f"Changed CIDR from {old_cidr} to {bad_cidr}",
    )


@register_tf_injector("SEMANTIC.invalid_region")
def inject_invalid_region(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Replace a valid region with a fake one."""
    m = re.search(r'(region\s*=\s*")([\w-]+)(")', text)
    if not m:
        return None

    old_region = m.group(2)
    bad_region = "us-fictional-1"
    modified = text[:m.start(2)] + bad_region + text[m.end(2):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_region,
        modified_snippet=bad_region,
        location=f"region at offset {m.start(2)}",
        description=f"Changed region from {old_region} to {bad_region}",
    )


# ---------------------------------------------------------------------------
# DEPENDENCY injectors
# ---------------------------------------------------------------------------

@register_tf_injector("DEPENDENCY.circular_dependency")
def inject_circular_dependency(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Add circular depends_on between two resources."""
    blocks = find_resource_blocks(text)
    if len(blocks) < 2:
        return None

    res1_type, res1_name, start1, end1 = blocks[0]
    res2_type, res2_name, start2, end2 = blocks[1]

    # Add depends_on to both resources pointing at each other
    lines = text.split('\n')
    # Find the closing brace of each block and insert depends_on before it
    block1_text = text[start1:end1]
    block2_text = text[start2:end2]

    last_brace1 = block1_text.rfind('}')
    last_brace2 = block2_text.rfind('}')

    dep1 = f'  depends_on = [{res2_type}.{res2_name}]'
    dep2 = f'  depends_on = [{res1_type}.{res1_name}]'

    # Insert deps (work backwards to preserve offsets)
    if start2 > start1:
        modified = (
            text[:start2 + last_brace2]
            + '\n' + dep2 + '\n'
            + text[start2 + last_brace2:start1 + last_brace1]
            if start1 + last_brace1 > start2 + last_brace2 else
            text[:start1 + last_brace1]
            + '\n' + dep1 + '\n'
            + text[start1 + last_brace1:start2 + last_brace2]
            + '\n' + dep2 + '\n'
            + text[start2 + last_brace2:]
        )
    else:
        modified = text

    # Simpler approach: just insert both depends_on
    modified = text[:start1 + last_brace1] + '\n' + dep1 + '\n' + text[start1 + last_brace1:]
    # Recalculate offset for second block
    offset_shift = len('\n' + dep1 + '\n')
    new_start2 = start2 + offset_shift if start2 > start1 else start2
    new_end2 = end2 + offset_shift if start2 > start1 else end2
    block2_in_modified = modified[new_start2:new_end2]
    last_brace2_new = block2_in_modified.rfind('}')
    modified = (
        modified[:new_start2 + last_brace2_new]
        + '\n' + dep2 + '\n'
        + modified[new_start2 + last_brace2_new:]
    )

    return modified, FaultInjection(
        fault_type=None,
        original_snippet="(no depends_on)",
        modified_snippet=f"{dep1}\n{dep2}",
        location=f"{res1_type}.{res1_name} <-> {res2_type}.{res2_name}",
        description=f"Added circular dependency between {res1_name} and {res2_name}",
    )


@register_tf_injector("DEPENDENCY.missing_depends_on")
def inject_missing_depends_on(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove an existing depends_on block."""
    m = re.search(r'(\n[ \t]*depends_on\s*=\s*\[[^\]]*\])', text)
    if not m:
        return None

    original = m.group(1)
    modified = text[:m.start()] + text[m.end():]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=original.strip(),
        modified_snippet="(removed)",
        location=f"offset {m.start()}",
        description="Removed depends_on declaration",
    )


# ---------------------------------------------------------------------------
# PROVIDER injectors
# ---------------------------------------------------------------------------

@register_tf_injector("PROVIDER.missing_provider")
def inject_missing_provider(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove the required_providers or terraform block."""
    terraform_blocks = find_block_boundaries(text, "terraform")
    if terraform_blocks:
        start, end = terraform_blocks[0]
        original = text[start:end]
        modified = text[:start] + text[end:]
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=original[:80] + "..." if len(original) > 80 else original,
            modified_snippet="(block removed)",
            location="terraform block",
            description="Removed terraform/required_providers block",
        )

    # Try removing provider block
    provider_blocks = find_block_boundaries(text, "provider")
    if provider_blocks:
        start, end = provider_blocks[0]
        original = text[start:end]
        modified = text[:start] + text[end:]
        return modified, FaultInjection(
            fault_type=None,
            original_snippet=original[:80] + "..." if len(original) > 80 else original,
            modified_snippet="(block removed)",
            location="provider block",
            description="Removed provider configuration block",
        )

    return None


@register_tf_injector("PROVIDER.version_constraint_mismatch")
def inject_version_constraint_mismatch(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Set an impossible provider version constraint."""
    m = re.search(r'(version\s*=\s*")([\s\S]*?)(")', text)
    if not m:
        # Try required_version
        m = re.search(r'(required_version\s*=\s*")([\s\S]*?)(")', text)
    if not m:
        return None

    old_version = m.group(2)
    impossible_version = ">= 99.0.0, < 99.0.1"
    modified = text[:m.start(2)] + impossible_version + text[m.end(2):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_version,
        modified_snippet=impossible_version,
        location=f"version constraint at offset {m.start(2)}",
        description=f"Changed version constraint to impossible range: {impossible_version}",
    )


# ---------------------------------------------------------------------------
# SECURITY injectors
# ---------------------------------------------------------------------------

@register_tf_injector("SECURITY.overly_permissive_security_group")
def inject_overly_permissive_sg(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Open a security group to 0.0.0.0/0 on all ports."""
    blocks = find_resource_blocks(text)
    sg_blocks = [(t, n, s, e) for t, n, s, e in blocks if t == "aws_security_group"]

    if not sg_blocks:
        return None

    res_type, res_name, start, end = sg_blocks[0]
    block_text = text[start:end]
    last_brace = block_text.rfind('}')

    open_ingress = """
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }"""

    modified = (
        text[:start + last_brace]
        + open_ingress + '\n'
        + text[start + last_brace:]
    )

    return modified, FaultInjection(
        fault_type=None,
        original_snippet="(security group without open ingress)",
        modified_snippet=open_ingress.strip(),
        location=f"resource \"aws_security_group\" \"{res_name}\"",
        description="Added overly permissive ingress rule (0.0.0.0/0 all TCP ports)",
    )


@register_tf_injector("SECURITY.missing_encryption")
def inject_missing_encryption(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Remove encryption configuration."""
    # Look for server_side_encryption_configuration, encrypted, kms_key_id, etc.
    encryption_patterns = [
        (r'\n[ \t]*server_side_encryption_configuration\s*\{[^}]*\{[^}]*\}[^}]*\}', "server_side_encryption_configuration block"),
        (r'\n[ \t]*encrypted\s*=\s*true', "encrypted = true"),
        (r'\n[ \t]*kms_key_id\s*=\s*"[^"]*"', "kms_key_id"),
        (r'\n[ \t]*storage_encrypted\s*=\s*true', "storage_encrypted = true"),
    ]

    for pattern, desc in encryption_patterns:
        m = re.search(pattern, text)
        if m:
            original = m.group(0).strip()
            modified = text[:m.start()] + text[m.end():]
            return modified, FaultInjection(
                fault_type=None,
                original_snippet=original[:80],
                modified_snippet="(removed)",
                location=f"encryption config",
                description=f"Removed {desc}",
            )

    return None


# ---------------------------------------------------------------------------
# CROSS_RESOURCE injectors
# ---------------------------------------------------------------------------

@register_tf_injector("CROSS_RESOURCE.subnet_vpc_mismatch")
def inject_subnet_vpc_mismatch(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Swap vpc_id or subnet_id references."""
    m = re.search(r'(vpc_id\s*=\s*)(\S+)', text)
    if not m:
        m = re.search(r'(subnet_id\s*=\s*)(\S+)', text)
    if not m:
        return None

    old_ref = m.group(2)
    fake_ref = '"vpc-00000000"' if 'vpc_id' in m.group(1) else '"subnet-00000000"'
    modified = text[:m.start(2)] + fake_ref + text[m.end(2):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=f"{m.group(1)}{old_ref}",
        modified_snippet=f"{m.group(1)}{fake_ref}",
        location=f"offset {m.start(2)}",
        description=f"Replaced {m.group(1).strip()} reference with hardcoded invalid ID",
    )


@register_tf_injector("CROSS_RESOURCE.wrong_az_reference")
def inject_wrong_az_reference(text: str, parsed: dict) -> tuple[str, FaultInjection] | None:
    """Change availability zone to a mismatched value."""
    m = re.search(r'(availability_zone\s*=\s*")([\w-]+)(")', text)
    if not m:
        return None

    old_az = m.group(2)
    # Pick a different AZ
    wrong_azs = ["ap-southeast-99a", "eu-fictional-1b", "us-nowhere-2c"]
    bad_az = random.choice(wrong_azs)
    modified = text[:m.start(2)] + bad_az + text[m.end(2):]

    return modified, FaultInjection(
        fault_type=None,
        original_snippet=old_az,
        modified_snippet=bad_az,
        location=f"availability_zone at offset {m.start(2)}",
        description=f"Changed availability zone from {old_az} to {bad_az}",
    )
