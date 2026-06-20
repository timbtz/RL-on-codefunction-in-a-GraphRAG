"""Unit tests for Phase 0.4: Sandbox.probe is per-query TOLERANT (post-mortem #3).

The run-7 probe aborted the whole candidate on the FIRST per-query crash (e.g.
`ValueError: ids is empty` on the fixed MAPK-kinase probe), *after* paying the
mutator -- 483K wasted tokens and crash-driven premature stops. The fix: run
every probe query, require >= min_ok successes, and only re-raise when too few
pass. The authoritative gate (evaluator.py) already tolerates per-query crashes.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_tolerant_probe
No FalkorDB / no network needed.
"""
import types

from graphretr_opt.env.sandbox import Sandbox, SandboxError


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _sandbox():
    # Sandbox.run only touches the graph via getattr-with-default, so a bare
    # namespace (no backend, no llm meter) is enough to exercise probe().
    return Sandbox(types.SimpleNamespace(), default_timeout_s=1.0)


def _fn_factory():
    """A candidate that returns a valid pred unless the query says 'empty', in
    which case it raises the exact run-7 error (a legitimately-empty intermediate
    on a hard query) -- the sandbox re-raises that as a per-query SandboxError."""
    def fn(q, G):
        if "empty" in q:
            raise ValueError("ids is empty")
        return {1: 1.0}
    return fn


def test_all_pass_returns_last_pred():
    sb, fn = _sandbox(), _fn_factory()
    out = sb.probe(fn, ["a", "b", "c"])
    _check("all-good probe returns a valid pred", out == {1: 1.0})


def test_one_bad_is_tolerated():
    sb, fn = _sandbox(), _fn_factory()
    # 2 good, 1 hard (empty) query -> min_ok=1 satisfied, must NOT raise.
    out = sb.probe(fn, ["good", "empty hard query", "good2"])
    _check("a single hard/empty probe query no longer aborts the candidate",
           out == {1: 1.0})


def test_all_bad_raises():
    sb, fn = _sandbox(), _fn_factory()
    try:
        sb.probe(fn, ["empty1", "empty2", "empty3"])
        _check("all-failing probe raises", False)
    except SandboxError as e:
        _check("all-failing probe raises SandboxError", "0/3" in str(e))


def test_min_ok_threshold():
    sb, fn = _sandbox(), _fn_factory()
    # only 1 of 3 succeeds; require 2 -> raise.
    try:
        sb.probe(fn, ["good", "empty", "empty"], min_ok=2)
        _check("min_ok=2 with 1 success raises", False)
    except SandboxError as e:
        _check("min_ok=2 with 1 success raises (1/3 < 2)", "1/3" in str(e))
    # require 1 -> pass.
    _check("min_ok=1 with 1 success passes",
           sb.probe(fn, ["good", "empty", "empty"], min_ok=1) == {1: 1.0})


def test_empty_probe_list():
    sb, fn = _sandbox(), _fn_factory()
    # no probe queries -> no-op pass (returns None), never raises.
    _check("empty probe list is a no-op pass (None)", sb.probe(fn, []) is None)


def main():
    test_all_pass_returns_last_pred()
    test_one_bad_is_tolerated()
    test_all_bad_raises()
    test_min_ok_threshold()
    test_empty_probe_list()
    print("\nall tolerant_probe tests passed")


if __name__ == "__main__":
    main()
