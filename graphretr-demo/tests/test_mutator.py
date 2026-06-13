"""Unit tests for the diff-based response protocol (optimizer/edits.py): parse,
apply, exact edit count, the full-module fallback, and malformed-block rejection.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_mutator
No FalkorDB / no network / no API key needed.
"""
from graphretr_opt.optimizer.edits import (
    EditError, apply_edits, count_edits, extract_code, parse_edit_blocks)

SRC = ("def search(q, G):\n"
       "    hits = G.vector_search(q, k=60)\n"
       "    return {i: s for i, s in hits}\n")


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def test_parse_and_apply():
    resp = ("Diagnosis: widen the pool.\n"
            "<<<<<<< SEARCH\n"
            "    hits = G.vector_search(q, k=60)\n"
            "=======\n"
            "    hits = G.vector_search(q, k=100)\n"
            ">>>>>>> REPLACE\n"
            "rationale: more candidates\n")
    blocks = parse_edit_blocks(resp)
    _check("parse: one block", count_edits(blocks) == 1)
    out = apply_edits(SRC, blocks)
    _check("apply: replacement took", "k=100" in out and "k=60" not in out)
    _check("apply: rest of module intact", "def search(q, G):" in out)


def test_multi_block_count_is_exact():
    resp = ("<<<<<<< SEARCH\n    hits = G.vector_search(q, k=60)\n=======\n"
            "    hits = G.vector_search(q, k=100)\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n    return {i: s for i, s in hits}\n=======\n"
            "    return {i: s * 2 for i, s in hits}\n>>>>>>> REPLACE\n")
    blocks = parse_edit_blocks(resp)
    _check("count: two blocks", count_edits(blocks) == 2)
    out = apply_edits(SRC, blocks)
    _check("apply: both blocks", "k=100" in out and "s * 2" in out)


def test_search_not_found():
    blocks = [("    hits = G.vector_search(q, k=999)\n", "    hits = []\n")]
    try:
        apply_edits(SRC, blocks)
        _check("not-found raises EditError", False)
    except EditError:
        _check("not-found raises EditError", True)


def test_ambiguous_match():
    src = "x = 1\nx = 1\n"
    try:
        apply_edits(src, [("x = 1", "x = 2")])
        _check("ambiguous match raises EditError", False)
    except EditError:
        _check("ambiguous match raises EditError", True)


def test_malformed_blocks_parse_empty():
    # missing the ======= divider -> not a block
    resp = "<<<<<<< SEARCH\nfoo\n>>>>>>> REPLACE\n"
    _check("malformed -> no blocks", parse_edit_blocks(resp) == [])


def test_full_module_fallback():
    resp = ("Here is the rewrite:\n```python\n" + SRC + "```\n")
    _check("fallback: no edit blocks", parse_edit_blocks(resp) == [])
    _check("fallback: extract_code returns module",
           "def search(q, G):" in extract_code(resp))
    try:
        extract_code("no code here")
        _check("fallback: empty raises", False)
    except ValueError:
        _check("fallback: empty raises", True)


def main():
    test_parse_and_apply()
    test_multi_block_count_is_exact()
    test_search_not_found()
    test_ambiguous_match()
    test_malformed_blocks_parse_empty()
    test_full_module_fallback()
    print("\nall mutator/edits tests passed")


if __name__ == "__main__":
    main()
