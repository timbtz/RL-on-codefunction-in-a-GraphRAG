"""edits.py -- the mutator's response protocols.

Two ways the optimizer agent may return a change, tried in this order:

  1. SEARCH/REPLACE blocks (Aider-style) -- the lean default. Each block names
     the exact incumbent lines to replace, so the edit budget is the literal
     block count (not a difflib estimate) and the reply carries only the changed
     regions instead of the whole module.
  2. A full ```python module -- the backward-compatible fallback for a model
     that ignores the block format; counted by difflib hunks at the call site.

Pure functions, no I/O -- unit-tested in tests/test_mutator.py.
"""
import re

_BLOCK = re.compile(
    r"<{3,}\s*SEARCH\s*\n(.*?)\n={3,}[ \t]*\n(.*?)\n>{3,}\s*REPLACE", re.S)


class EditError(ValueError):
    """A SEARCH/REPLACE block could not be applied (search text absent or
    ambiguous). The caller falls back to the full-module path."""


def parse_edit_blocks(text):
    """-> [(search, replace)] for every well-formed block ([] if none)."""
    return [(m.group(1), m.group(2)) for m in _BLOCK.finditer(text)]


def apply_edits(src, blocks):
    """Apply blocks in order. Each SEARCH must match the *current* source
    exactly once (anchored, unambiguous). -> new source. Raises EditError."""
    out = src
    for search, replace in blocks:
        if not search.strip():
            raise EditError("empty SEARCH block")
        n = out.count(search)
        if n != 1:
            raise EditError(
                f"SEARCH block matches {n} sites (need exactly 1): {search[:60]!r}")
        out = out.replace(search, replace, 1)
    return out if out.endswith("\n") else out + "\n"


def count_edits(blocks):
    return len(blocks)


def extract_code(text):
    """Last ```python fenced block -- the full-module fallback."""
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no fenced code block in LLM response")
    return blocks[-1].strip() + "\n"
