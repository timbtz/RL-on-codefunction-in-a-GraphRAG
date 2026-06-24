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

# Multi-file variant (graph_search target): an OPTIONAL `FILE: <relpath>` header
# precedes the same SEARCH/REPLACE body, so one reply can edit several files.
# A header-less block keeps the single-file meaning (relpath -> None), which the
# multi-file applier resolves to the lone default editable file -- so the format
# is a strict superset of the single-file one (back-compat).
_BLOCK_FILE = re.compile(
    r"(?:FILE:[ \t]*([^\n]+)\n)?"
    r"<{3,}\s*SEARCH\s*\n(.*?)\n={3,}[ \t]*\n(.*?)\n>{3,}\s*REPLACE", re.S)


class EditError(ValueError):
    """A SEARCH/REPLACE block could not be applied (search text absent or
    ambiguous). The caller falls back to the full-module path."""


def parse_edit_blocks(text):
    """-> [(search, replace)] for every well-formed block ([] if none)."""
    return [(m.group(1), m.group(2)) for m in _BLOCK.finditer(text)]


def parse_edit_blocks_multi(text):
    """Multi-file parse -> [(relpath_or_None, search, replace)] for every
    well-formed block ([] if none). relpath is the FILE: header stripped of
    surrounding whitespace, or None when the block carries no header."""
    out = []
    for m in _BLOCK_FILE.finditer(text):
        rel = m.group(1)
        out.append((rel.strip() if rel else None, m.group(2), m.group(3)))
    return out


def apply_edits_multi(overlay, blocks, editable=None, default_file=None):
    """Apply [(relpath_or_None, search, replace)] blocks to a {relpath: src}
    overlay, returning a NEW overlay (the input is not mutated).

    - A block with no FILE: header targets `default_file` (the lone editable
      file when there is exactly one) -- this is the single-file back-compat path.
    - Each SEARCH must match the *current* source of its target file exactly once
      (the per-file anchor invariant).
    - A block naming a file outside `editable` is rejected (the mutator may only
      touch the declared editable set).
    Raises EditError (with the file name) on any violation."""
    if default_file is None and editable and len(editable) == 1:
        default_file = editable[0]
    editable_set = set(editable) if editable is not None else None
    out = dict(overlay)
    for rel, search, replace in blocks:
        target = rel or default_file
        if target is None:
            raise EditError("block has no FILE: header and there is no single "
                            "default editable file to apply it to")
        target = target.replace("\\", "/").lstrip("/")
        if editable_set is not None and target not in editable_set:
            raise EditError(f"FILE {target!r} is not in the editable set "
                            f"{sorted(editable_set)}")
        if target not in out:
            raise EditError(f"FILE {target!r} is not part of the overlay")
        if not search.strip():
            raise EditError(f"empty SEARCH block for {target!r}")
        n = out[target].count(search)
        if n != 1:
            raise EditError(
                f"SEARCH block for {target!r} matches {n} sites (need exactly "
                f"1): {search[:60]!r}")
        new_src = out[target].replace(search, replace, 1)
        out[target] = new_src if new_src.endswith("\n") else new_src + "\n"
    return out


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
