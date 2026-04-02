"""HCL text manipulation helpers for Terraform fault injection.

python-hcl2 is read-only (parses HCL to dicts but can't write back),
so we use a parse-then-regex approach: parse to understand structure,
then do targeted string manipulation on the raw text.
"""

from __future__ import annotations

import re


def find_block_boundaries(
    text: str, block_type: str, block_name: str | None = None
) -> list[tuple[int, int]]:
    """Find start/end character offsets of HCL blocks by brace-depth counting.

    Args:
        text: Raw HCL text.
        block_type: e.g. "resource", "variable", "provider", "terraform".
        block_name: Optional label to match (e.g. "aws_instance" or "\"main\"").

    Returns:
        List of (start_offset, end_offset) tuples for matching blocks.
    """
    results = []
    # Match block headers like: resource "aws_instance" "main" {
    if block_name:
        pattern = re.compile(
            rf'^[ \t]*{re.escape(block_type)}\s+["\']?{re.escape(block_name)}["\']?'
            r'(?:\s+["\'][^"\']*["\'])?\s*\{',
            re.MULTILINE,
        )
    else:
        pattern = re.compile(
            rf'^[ \t]*{re.escape(block_type)}\s+.*?\{{',
            re.MULTILINE,
        )

    for match in pattern.finditer(text):
        start = match.start()
        brace_pos = match.end() - 1  # Position of opening brace
        depth = 1
        pos = brace_pos + 1

        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == '"':
                # Skip string content
                pos += 1
                while pos < len(text) and text[pos] != '"':
                    if text[pos] == '\\':
                        pos += 1
                    pos += 1
            elif ch == '#':
                # Skip line comment
                while pos < len(text) and text[pos] != '\n':
                    pos += 1
            pos += 1

        if depth == 0:
            results.append((start, pos))

    return results


def find_attribute_line(
    text: str, block_start: int, block_end: int, attr_name: str
) -> int | None:
    """Find the line number of a specific attribute assignment within a block.

    Returns 0-based line number or None if not found.
    """
    block_text = text[block_start:block_end]
    lines = text[:block_start].count('\n')

    for i, line in enumerate(block_text.split('\n')):
        stripped = line.strip()
        # Match attr = value or attr= value patterns
        if re.match(rf'{re.escape(attr_name)}\s*=', stripped):
            return lines + i

    return None


def remove_lines(text: str, start_line: int, end_line: int) -> str:
    """Remove line range [start_line, end_line] (0-based, inclusive)."""
    lines = text.split('\n')
    result = lines[:start_line] + lines[end_line + 1:]
    return '\n'.join(result)


def replace_value(text: str, line_num: int, old_val: str, new_val: str) -> str:
    """Replace a value on a specific line (0-based)."""
    lines = text.split('\n')
    if 0 <= line_num < len(lines):
        lines[line_num] = lines[line_num].replace(old_val, new_val, 1)
    return '\n'.join(lines)


def find_all_attributes(text: str, block_start: int, block_end: int) -> list[tuple[str, int]]:
    """Find all attribute assignments within a block.

    Returns list of (attr_name, line_number) tuples.
    """
    block_text = text[block_start:block_end]
    base_line = text[:block_start].count('\n')
    attrs = []
    depth = 0

    for i, line in enumerate(block_text.split('\n')):
        stripped = line.strip()
        depth += stripped.count('{') - stripped.count('}')
        if depth <= 1:  # Only top-level attributes of this block
            m = re.match(r'(\w+)\s*=', stripped)
            if m:
                attrs.append((m.group(1), base_line + i))

    return attrs


def find_resource_blocks(text: str) -> list[tuple[str, str, int, int]]:
    """Find all resource blocks and return (type, name, start, end) tuples."""
    results = []
    pattern = re.compile(
        r'^[ \t]*resource\s+"([^"]+)"\s+"([^"]+)"\s*\{',
        re.MULTILINE,
    )

    for match in pattern.finditer(text):
        res_type = match.group(1)
        res_name = match.group(2)
        start = match.start()
        brace_pos = match.end() - 1
        depth = 1
        pos = brace_pos + 1

        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == '"':
                pos += 1
                while pos < len(text) and text[pos] != '"':
                    if text[pos] == '\\':
                        pos += 1
                    pos += 1
            pos += 1

        if depth == 0:
            results.append((res_type, res_name, start, pos))

    return results


def find_variable_refs(text: str) -> list[tuple[str, int]]:
    """Find all var.X references and return (var_name, offset) pairs."""
    results = []
    for m in re.finditer(r'var\.(\w+)', text):
        results.append((m.group(1), m.start()))
    return results


def find_resource_refs(text: str) -> list[tuple[str, str, int]]:
    """Find all resource_type.resource_name references.

    Returns (resource_type, resource_name, offset) triples.
    """
    results = []
    # Match patterns like aws_instance.main.id or aws_vpc.default.id
    for m in re.finditer(r'(\w+\.\w+)\.(\w+)', text):
        full_ref = m.group(1)
        parts = full_ref.split('.')
        if len(parts) == 2 and not parts[0].startswith('var'):
            results.append((parts[0], parts[1], m.start()))
    return results
