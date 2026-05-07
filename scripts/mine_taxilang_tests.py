"""Mine taxilang Kotlin test suite for (broken_taxi, expected_error) pairs and valid Taxi snippets.

Looks for two patterns in compiler/src/test/**/*.kt:

  Broken pair:
    \"\"\"
    <taxi source>
    \"\"\"[.trimIndent()]
       .validated()
       .shouldContainMessage[StartingWith]("<error message>")

  Valid snippet:
    \"\"\"
    <taxi source>
    \"\"\"[.trimIndent()]
       .compiled()        # without a following shouldContainMessage in the chain

Multiple .shouldContainMessage calls chained on the same source produce one
record with multiple errors.

Outputs:
  data/taxi/inversion_pairs.jsonl   { broken, errors[], assertion, source_file, source_line }
  data/taxi/valid_corpus.jsonl       { taxi, source_file, source_line }
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO / "data/upstream/taxilang/compiler/src/test"
OUT_DIR = REPO / "data/taxi"

TRIPLE = '"""'

# After a triple-quoted string we may see arbitrary chained calls.
# We capture the chain following the closing """ until the first non-chain token.
CHAIN_CALL = re.compile(
    r"""
    \s*\.\s*
    (?P<call>[A-Za-z_][A-Za-z0-9_]*)      # method name
    \s*
    (?:\(                                 # optional argument list
        (?P<args>(?:[^()"]|"(?:[^"\\]|\\.)*")*)
       \))?
    """,
    re.VERBOSE,
)

ERROR_ASSERTIONS = {"shouldContainMessage", "shouldContainMessageStartingWith"}
COMPILE_TERMINALS = {"compiled", "validated"}


def kotlin_trim_indent(s: str) -> str:
    """Reproduce Kotlin's String.trimIndent() reasonably faithfully."""
    lines = s.split("\n")
    # drop leading/trailing blank lines
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        return ""
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    common = min(indents) if indents else 0
    return "\n".join(ln[common:] if len(ln) >= common else ln for ln in lines)


def _find_triples(text: str) -> list[tuple[int, int, str]]:
    """Return (open_idx, close_idx_after, content) for each \"\"\"...\"\"\" block."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find(TRIPLE, i)
        if j == -1:
            break
        k = text.find(TRIPLE, j + 3)
        if k == -1:
            break
        out.append((j, k + 3, text[j + 3 : k]))
        i = k + 3
    return out


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _parse_chain(text: str, start: int) -> list[tuple[str, str | None]]:
    """Greedy parse of '.method(args)' calls starting at `start`.

    Returns [(method_name, raw_args_or_None), ...]. Stops at the first character
    that doesn't fit the chain shape.
    """
    calls: list[tuple[str, str | None]] = []
    i = start
    n = len(text)
    while i < n:
        # skip whitespace
        while i < n and text[i] in " \t\n\r":
            i += 1
        if i >= n or text[i] != ".":
            break
        # match `.name`
        j = i + 1
        while j < n and (text[j].isalnum() or text[j] == "_"):
            j += 1
        if j == i + 1:
            break
        name = text[i + 1 : j]
        # optional args (we hand-parse to handle nested parens / quoted strings)
        args: str | None = None
        k = j
        while k < n and text[k] in " \t\n\r":
            k += 1
        if k < n and text[k] == "(":
            depth = 1
            m = k + 1
            in_str: str | None = None
            while m < n and depth > 0:
                c = text[m]
                if in_str is not None:
                    if c == "\\" and m + 1 < n:
                        m += 2
                        continue
                    if c == in_str:
                        in_str = None
                elif c in ('"', "'"):
                    in_str = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                m += 1
            if depth != 0:
                break
            args = text[k + 1 : m - 1]
            calls.append((name, args))
            i = m
        else:
            calls.append((name, None))
            i = j
    return calls


_DOUBLE_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _first_string_arg(args: str) -> str | None:
    """Extract the first double-quoted string literal from a Kotlin arg list."""
    m = _DOUBLE_QUOTED.search(args)
    if not m:
        return None
    raw = m.group(1)
    # decode common Kotlin escapes
    return (
        raw.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", '"')
        .replace("\\\\", "\\")
    )


_VAL_DECL = re.compile(r"\bval\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")


def _record(triple_content: str, applied_trim: bool) -> str:
    return kotlin_trim_indent(triple_content) if applied_trim else triple_content.strip("\n")


def _find_var_assignments(text: str, triples: list[tuple[int, int, str]]) -> dict[int, str]:
    """For each triple, check if it's the RHS of `val NAME = "..."[.trimIndent()]`.

    Returns {triple_index_in_list: var_name}.
    """
    out: dict[int, str] = {}
    for idx, (open_idx, _, _) in enumerate(triples):
        # look back from open_idx to find `val NAME = `
        head = text[max(0, open_idx - 200) : open_idx]
        m = list(_VAL_DECL.finditer(head))
        if not m:
            continue
        last = m[-1]
        # ensure nothing other than whitespace sits between the assignment and the """
        between = head[last.end() :]
        if between.strip() == "":
            out[idx] = last.group(1)
    return out


def _scan_var_usages(
    text: str, var_name: str, after_idx: int, scope_end: int | None = None
) -> list[tuple[int, list[tuple[str, str | None]]]]:
    """Find usages `<var>.<chain>` after position after_idx, bounded by scope_end."""
    out = []
    pat = re.compile(rf"\b{re.escape(var_name)}\b")
    end_limit = scope_end if scope_end is not None else len(text)
    i = after_idx
    while i < end_limit:
        m = pat.search(text, i, end_limit)
        if not m:
            break
        end = m.end()
        chain = _parse_chain(text, end)
        if chain:
            out.append((m.start(), chain))
        i = end
    return out


def _next_rebinding_pos(text: str, var_name: str, after_idx: int) -> int:
    """Return position where this variable name is re-declared (`val NAME = `) again,
    or len(text) if it isn't. Caps the scope for usage scanning so that two
    consecutive tests both using `val source = "..."` don't cross-contaminate."""
    pat = re.compile(rf"\bval\s+{re.escape(var_name)}\s*=", re.MULTILINE)
    m = pat.search(text, after_idx)
    return m.start() if m else len(text)


# Recognises chains where the FIRST call's first arg is itself a Taxi source string
# (a triple-quoted Kotlin block). When the receiver triple is `s1` and the argument
# triple is `s2`, the effective compile unit is `s1 + "\n" + s2`. The error
# assertion typically arrives later via `.errors.shouldContainMessage(...)`.
MULTI_SOURCE_TAXI_CALLS = {
    "compiledWithQuery",
    "compiledWithQueryProducingCompilationException",
    "compiledWithQueryProducingCompilationErrors",
    "validatedWithQuery",
    "compiledForStrings",
    "concatenatedWith",
}


def _arg_is_triple_quoted(args: str) -> str | None:
    """If the args string contains a Kotlin triple-quoted block, return its content."""
    j = args.find(TRIPLE)
    if j == -1:
        return None
    k = args.find(TRIPLE, j + 3)
    if k == -1:
        return None
    return args[j + 3 : k]


def mine_file(path: Path) -> tuple[list[dict], list[dict]]:
    text = path.read_text(errors="replace")
    rel = path.relative_to(REPO)

    broken: list[dict] = []
    valid: list[dict] = []

    triples = _find_triples(text)
    var_for_triple = _find_var_assignments(text, triples)

    # set to skip triples that have been consumed as the SECOND source in a
    # multi-source compile pattern (so they aren't double-emitted as standalone)
    consumed: set[int] = set()
    triple_starts = {open_idx: i for i, (open_idx, _, _) in enumerate(triples)}

    for idx, (open_idx, after_idx, content) in enumerate(triples):
        if idx in consumed:
            continue
        line = _line_of(text, open_idx)

        # --- direct chain on the triple-quoted literal ---
        direct_chain = _parse_chain(text, after_idx)
        applied_trim_direct = any(name == "trimIndent" for name, _ in direct_chain)

        # --- if assigned to a var, also gather chains from each usage ---
        # Bound the scope at the var's next re-binding to avoid cross-contamination.
        var_name = var_for_triple.get(idx)
        usages: list[list[tuple[str, str | None]]] = []
        if var_name:
            scope_end = _next_rebinding_pos(text, var_name, after_idx)
            for _, ch in _scan_var_usages(text, var_name, after_idx, scope_end):
                usages.append(ch)

        # combine for terminal/assertion analysis
        all_chains: list[list[tuple[str, str | None]]] = [direct_chain] + usages

        # --- multi-source compile detection ---
        # If the chain contains a `.compiledWithQuery*(<triple>)` style call,
        # treat receiver + that arg-triple as a single compile unit. Mark the
        # arg-triple as consumed so we don't emit it standalone.
        extra_source_parts: list[str] = []
        for chain in all_chains:
            for name, args in chain:
                if name in MULTI_SOURCE_TAXI_CALLS and args:
                    extra = _arg_is_triple_quoted(args)
                    if extra is not None:
                        extra_source_parts.append(extra)
                        # find which triple this corresponds to and mark consumed
                        # search for the inner triple's open position in the file
                        # (the arg string is a substring of `text` after `(`)
                        # — we identify by content match against any later triple
                        for j_idx in range(idx + 1, len(triples)):
                            if triples[j_idx][2] == extra:
                                consumed.add(j_idx)
                                break

        applied_trim = applied_trim_direct or any(
            n == "trimIndent" for ch in usages for n, _ in ch
        )
        taxi_src = _record(content, applied_trim)
        if extra_source_parts:
            extras = "\n".join(_record(p, applied_trim) for p in extra_source_parts)
            taxi_src = f"{taxi_src}\n{extras}"

        # collect every error assertion across direct + usage chains
        error_calls: list[tuple[str, str | None]] = []
        terminal_kinds: set[str] = set()
        for ch in all_chains:
            for name, args in ch:
                if name in ERROR_ASSERTIONS:
                    error_calls.append((name, args))
                if name in COMPILE_TERMINALS:
                    terminal_kinds.add(name)
                # `Compiler(src).compile()` patterns mean the source is intended valid;
                # mark with a synthetic terminal to feed valid-corpus
                if name == "compile":
                    terminal_kinds.add("compiled")

        if error_calls:
            errors = []
            for name, args in error_calls:
                if not args:
                    continue
                msg = _first_string_arg(args)
                if msg is None:
                    continue
                errors.append({"assertion": name, "message": msg})
            if not errors:
                continue
            broken.append(
                {
                    "broken": taxi_src,
                    "errors": errors,
                    "source_file": str(rel),
                    "source_line": line,
                    "via_var": var_name,
                }
            )
        elif "compiled" in terminal_kinds:
            valid.append(
                {
                    "taxi": taxi_src,
                    "source_file": str(rel),
                    "source_line": line,
                    "via_var": var_name,
                }
            )

    return broken, valid


def main() -> None:
    if not TEST_ROOT.exists():
        raise SystemExit(f"taxilang test root not found: {TEST_ROOT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_broken: list[dict] = []
    all_valid: list[dict] = []
    for kt in sorted(TEST_ROOT.rglob("*.kt")):
        b, v = mine_file(kt)
        all_broken.extend(b)
        all_valid.extend(v)

    inv_path = OUT_DIR / "inversion_pairs.jsonl"
    with inv_path.open("w") as f:
        for r in all_broken:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    val_path = OUT_DIR / "valid_corpus.jsonl"
    with val_path.open("w") as f:
        for r in all_valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"inversion_pairs:  {len(all_broken):4d} -> {inv_path}")
    print(f"valid_corpus:     {len(all_valid):4d} -> {val_path}")

    # quick stats
    files = {r["source_file"] for r in all_broken} | {r["source_file"] for r in all_valid}
    print(f"source files covered: {len(files)}")
    multi_err = [r for r in all_broken if len(r["errors"]) > 1]
    print(f"pairs with multiple expected messages: {len(multi_err)}")


if __name__ == "__main__":
    main()
