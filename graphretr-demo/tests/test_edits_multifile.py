"""Unit tests for multi-file SEARCH/REPLACE (optimizer/edits.py): FILE: header
parsing, per-file exactly-once anchor, editable-set enforcement, and the
header-less back-compat path.
"""
import pytest

from graphretr_opt.optimizer.edits import (
    EditError, apply_edits_multi, parse_edit_blocks_multi)

A = "common/a.py"
B = "common/b.py"
OVERLAY = {A: "x = 1\ny = 2\n", B: "z = 3\n"}


def test_parse_with_and_without_header():
    resp = (
        f"FILE: {A}\n<<<<<<< SEARCH\nx = 1\n=======\nx = 11\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nz = 3\n=======\nz = 33\n>>>>>>> REPLACE\n")
    blocks = parse_edit_blocks_multi(resp)
    assert len(blocks) == 2
    assert blocks[0][0] == A          # header captured
    assert blocks[1][0] is None       # header-less


def test_apply_multi_targets_named_files():
    resp = (f"FILE: {A}\n<<<<<<< SEARCH\nx = 1\n=======\nx = 11\n>>>>>>> REPLACE\n"
            f"FILE: {B}\n<<<<<<< SEARCH\nz = 3\n=======\nz = 33\n>>>>>>> REPLACE\n")
    blocks = parse_edit_blocks_multi(resp)
    out = apply_edits_multi(OVERLAY, blocks, editable=(A, B))
    assert out[A] == "x = 11\ny = 2\n"
    assert out[B] == "z = 33\n"
    assert OVERLAY[A] == "x = 1\ny = 2\n"   # input not mutated


def test_headerless_defaults_to_single_editable():
    resp = "<<<<<<< SEARCH\nx = 1\n=======\nx = 9\n>>>>>>> REPLACE\n"
    blocks = parse_edit_blocks_multi(resp)
    out = apply_edits_multi({A: "x = 1\n"}, blocks, editable=(A,))
    assert out[A] == "x = 9\n"


def test_reject_non_editable_file():
    resp = f"FILE: {B}\n<<<<<<< SEARCH\nz = 3\n=======\nz = 4\n>>>>>>> REPLACE\n"
    blocks = parse_edit_blocks_multi(resp)
    with pytest.raises(EditError):
        apply_edits_multi(OVERLAY, blocks, editable=(A,))  # B not editable


def test_anchor_must_match_exactly_once():
    overlay = {A: "x = 1\nx = 1\n"}
    blocks = [(A, "x = 1", "x = 2")]
    with pytest.raises(EditError):
        apply_edits_multi(overlay, blocks, editable=(A,))


def test_missing_search_raises():
    blocks = [(A, "nonexistent", "y")]
    with pytest.raises(EditError):
        apply_edits_multi(OVERLAY, blocks, editable=(A, B))
