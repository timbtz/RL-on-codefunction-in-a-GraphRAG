"""Unit tests for Item 1: compiler-in-the-loop self-repair in Mutator.propose.

A candidate that PARSES and fits the edit budget but FAILS the sandbox/probe is
fed its exact error back into the same conversation for a targeted fix, up to
`repair_budget` times -- the KernelEvolve read/replace/lint pattern. With
repair_budget=0 the candidate is still validated (so the loop's fn_cache is
populated) but a probe failure is a hard reject; with validate=None the
validation arm never runs and propose returns the unchecked candidate.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_self_repair
No FalkorDB / no network / no API key needed (agent + validate are stubbed).
"""
from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.env.errors import SAFE_BUILTINS, SandboxError
from graphretr_opt.optimizer.mutator import Mutator


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


SEED = SearchProgram("def search(q, G):\n    return {0: 1.0}\n", family="t")


def _failure():
    return [{"query": "q", "bucket": "missed",
             "metrics": {"recall@20": 0.3, "hit@1": 0.0, "hit@5": 0.0, "mrr": 0.1},
             "attribution": "GENERATION (gold not in top-100, ...)",
             "empty_result": False, "missed_gold": [], "top_wrong": [],
             "gold_ranks": [], "error": None}]


class _ScriptedAgent:
    """Returns the i-th canned ```python module per complete() call, recording
    every prompt so a test can assert the repair error was injected."""

    def __init__(self, replies):
        self._replies = replies
        self.calls = 0
        self.prompts = []

    def digest(self, evidence):
        return evidence

    def complete(self, prompt, tier="editor"):
        self.prompts.append(prompt)
        i = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        return f"```python\n{self._replies[i]}```"


_BAD = "def search(q, G):\n    return {1: 1.0}  # BAD\n"
_GOOD = "def search(q, G):\n    return {2: 2.0}  # GOOD\n"


def _validate_rejecting_bad(cand):
    if "BAD" in cand.src:
        raise SandboxError("probe failed: 0/3 probe queries returned a valid pred")


def test_repair_succeeds_on_second_attempt():
    agent = _ScriptedAgent([_BAD, _GOOD])
    mut = Mutator(agent, "PRIMITIVES-DOC", SAFE_BUILTINS)
    cand, transcript, meta = mut.propose(
        SEED, _failure(), [], [], edit_budget=20,
        validate=_validate_rejecting_bad, repair_budget=1)
    _check("self-repair: returns the repaired (GOOD) candidate",
           cand is not None and "GOOD" in cand.src)
    _check("self-repair: probe_failed counts the one rejection",
           meta["probe_failed"] == 1)
    _check("self-repair: no reject_reason on success",
           meta["reject_reason"] is None)
    _check("self-repair: exactly two agent calls (initial + one repair)",
           agent.calls == 2)
    _check("self-repair: the exact probe error was fed back into the prompt",
           "probe failed: 0/3" in agent.prompts[1])


def test_repair_budget_zero_validates_and_rejects():
    agent = _ScriptedAgent([_BAD])
    mut = Mutator(agent, "PRIMITIVES-DOC", SAFE_BUILTINS)
    cand, _, meta = mut.propose(
        SEED, _failure(), [], [], edit_budget=20,
        validate=_validate_rejecting_bad, repair_budget=0)
    _check("repair_budget=0: a probe-failing candidate is rejected",
           cand is None and meta["reject_reason"] == "sandbox")
    _check("repair_budget=0: still validated once (probe_failed==1)",
           meta["probe_failed"] == 1)
    _check("repair_budget=0: no repair attempt (one agent call)",
           agent.calls == 1)


def test_repair_exhausted_after_budget():
    agent = _ScriptedAgent([_BAD, _BAD, _BAD])
    mut = Mutator(agent, "PRIMITIVES-DOC", SAFE_BUILTINS)
    cand, _, meta = mut.propose(
        SEED, _failure(), [], [], edit_budget=20,
        validate=_validate_rejecting_bad, repair_budget=1)
    _check("repair exhausted: rejected with reason 'sandbox'",
           cand is None and meta["reject_reason"] == "sandbox")
    _check("repair exhausted: both probe failures counted",
           meta["probe_failed"] == 2)
    _check("repair exhausted: initial + one repair = two calls",
           agent.calls == 2)


def test_no_validate_returns_unchecked():
    agent = _ScriptedAgent([_BAD])
    mut = Mutator(agent, "PRIMITIVES-DOC", SAFE_BUILTINS)
    cand, _, meta = mut.propose(SEED, _failure(), [], [], edit_budget=20)
    _check("validate=None: candidate returned unchecked (old behavior)",
           cand is not None and "BAD" in cand.src)
    _check("validate=None: probe_failed stays 0 and reason is None",
           meta["probe_failed"] == 0 and meta["reject_reason"] is None)


def main():
    test_repair_succeeds_on_second_attempt()
    test_repair_budget_zero_validates_and_rejects()
    test_repair_exhausted_after_budget()
    test_no_validate_returns_unchecked()
    print("\nall self-repair tests passed")


if __name__ == "__main__":
    main()
