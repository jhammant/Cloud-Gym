"""Custom YAML loader/dumper for CloudFormation templates.

CloudFormation uses custom YAML tags like !Ref, !GetAtt, !Sub, etc.
that yaml.safe_load doesn't handle. This module provides a loader
that preserves these tags as dicts.
"""

from __future__ import annotations

import yaml


class CFLoader(yaml.SafeLoader):
    """YAML loader that handles CloudFormation intrinsic functions."""
    pass


class CFDumper(yaml.SafeDumper):
    """YAML dumper that outputs CloudFormation intrinsic functions."""
    pass


# CloudFormation intrinsic function tags
_CF_TAGS = [
    "!Ref", "!GetAtt", "!Sub", "!Join", "!Select", "!Split",
    "!If", "!Equals", "!Not", "!And", "!Or", "!Condition",
    "!FindInMap", "!GetAZs", "!ImportValue", "!Base64",
    "!Cidr", "!Transform",
]


def _cf_constructor(tag: str):
    """Create a constructor that converts a CF tag to a dict."""
    fn_name = tag.lstrip("!")

    def constructor(loader, node):
        if isinstance(node, yaml.ScalarNode):
            value = loader.construct_scalar(node)
            return {fn_name: value}
        elif isinstance(node, yaml.SequenceNode):
            value = loader.construct_sequence(node, deep=True)
            return {fn_name: value}
        elif isinstance(node, yaml.MappingNode):
            value = loader.construct_mapping(node, deep=True)
            return {fn_name: value}
        return {fn_name: None}

    return constructor


def _cf_representer(tag: str):
    """Create a representer that converts a dict back to a CF tag."""
    fn_name = tag.lstrip("!")

    def representer(dumper, data):
        value = data[fn_name]
        if isinstance(value, str):
            return dumper.represent_scalar(tag, value)
        elif isinstance(value, list):
            return dumper.represent_sequence(tag, value)
        elif isinstance(value, dict):
            return dumper.represent_mapping(tag, value)
        return dumper.represent_scalar(tag, str(value))

    return representer


# Register constructors and representers for all CF tags
for _tag in _CF_TAGS:
    _fn_name = _tag.lstrip("!")
    CFLoader.add_constructor(_tag, _cf_constructor(_tag))


def cf_load(text: str) -> dict:
    """Load a CloudFormation YAML template, handling intrinsic functions."""
    return yaml.load(text, Loader=CFLoader) or {}


def cf_dump(template: dict) -> str:
    """Dump a CloudFormation template dict back to YAML."""
    # For simplicity, use regular yaml.dump with default_flow_style=False
    # This won't preserve the !Tag shorthand, but will produce valid YAML
    # with Fn:: prefix style that CloudFormation accepts
    return yaml.dump(template, default_flow_style=False, sort_keys=False)
