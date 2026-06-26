"""Family-aware STaRK mutation-prompt tests (run11 fix).

Asserts the FileSet mutation prompt is target-correct for the stark_search path
(STaRK-prime biomedical retrieval) and byte-unchanged for the graph_search path,
plus the family-selected efficiency legend and the rollout-aggregate header.

NOTE on the cost-shy phrase: the source legend (mutator.py `_EFFICIENCY_LEGEND`)
spells it lowercase -- "do not pile on graph hops" -- so these assertions use that
exact casing (the plan's draft used "do NOT" which does not match the constant).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_stark_prompt
No FalkorDB / no network / no API key needed -- pure prompt-string assembly.
"""
from graphretr_opt.optimizer import mutator
from graphretr_opt.optimizer.mutator import build_search_prompt, format_evidence
from graphretr_opt.artifact.file_set import FileSet

_OVERLAY = {"svc/search.py": "class S:\n    def search(self, query):\n        return {}\n"}
_EDITABLE = ("svc/search.py",)


def _fs(family):
    # Minimal FileSet -- build_search_prompt only reads .family / .editable /
    # .overlay (see Task 6 GOTCHA in the plan); no real base checkout needed.
    return FileSet("/tmp/base", dict(_OVERLAY), _EDITABLE, family=family)


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def test_stark_prompt_is_target_correct():
    p = build_search_prompt(_fs("stark_search"), "DOMAINNOTE", "DIGEST", 4)
    for needle in ("STaRK-prime", "recall@20", "dict[int, float]"):
        _check(f"stark contains {needle!r}", needle in p)
    for bad in ("German company", "multiple-choice", "frozen", "FROZEN",
                "closed-book"):
        _check(f"stark excludes {bad!r}", bad not in p)
    _check("stark drops the inert cost bullet", "Cheaper retrieval" not in p)


def test_graph_search_prompt_unchanged():
    p = build_search_prompt(_fs("graph_search"), "DOMAINNOTE", "DIGEST", 4)
    _check("graph_search still German KG", "German company" in p)
    _check("graph_search still MCQ", "multiple-choice" in p)
    _check("graph_search keeps cost bullet", "Cheaper retrieval" in p)


def test_unknown_family_falls_back_to_graph_search():
    # unknown family -> _SEARCH_INTRO default + graph_search cost bullet (no crash)
    p = build_search_prompt(_fs("reasoning_first_v7"), "DOMAINNOTE", "DIGEST", 4)
    _check("unknown family uses graph_search framing", "German company" in p)
    _check("unknown family keeps cost bullet", "Cheaper retrieval" in p)


def test_legend_is_family_selected():
    stark = format_evidence([], [], [], legend=mutator._STARK_LEGEND)
    default = format_evidence([], [], [])
    _check("stark legend drops cost-shy phrase",
           "do not pile on graph hops" not in stark)
    _check("default legend keeps cost-shy phrase (graph_search)",
           "do not pile on graph hops" in default)


def test_summary_header_is_prepended():
    e = format_evidence([], [], [], summary="## Rollout summary: X")
    _check("summary header rendered", "Rollout summary" in e)
    _check("summary precedes the legend/failures",
           e.startswith("## Rollout summary"))


def test_stark_combine_mode_keeps_intro_and_combine_block():
    a = _fs("stark_search")
    b = _fs("stark_search")
    p = build_search_prompt(a, "DOMAINNOTE", "DIGEST", 4, mate=b)
    _check("combine: stark intro retained", "STaRK-prime" in p)
    _check("combine: candidate B block present", "candidate B" in p)


def main():
    test_stark_prompt_is_target_correct()
    test_graph_search_prompt_unchanged()
    test_unknown_family_falls_back_to_graph_search()
    test_legend_is_family_selected()
    test_summary_header_is_prepended()
    test_stark_combine_mode_keeps_intro_and_combine_block()
    print("\nall stark-prompt tests passed")


if __name__ == "__main__":
    main()
