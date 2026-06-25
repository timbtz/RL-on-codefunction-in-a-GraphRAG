"""Unit tests for Phase D1: the TieredCoder routing (Haiku analyst digests,
Sonnet editor edits off-plateau, Opus architect on-plateau) and the per-tier
call ledger, plus Mutator's plateau->architect tier selection.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_tiered
No FalkorDB / no network / no CLI needed (sub-agents are stubbed).
"""
from collections import Counter

from graphretr_opt.agents.team import TieredCoder, _ANALYST
from graphretr_opt.optimizer.mutator import Mutator
from graphretr_opt.env.errors import SAFE_BUILTINS


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


class _FakeCoder:
    """Stands in for a SingleCoder tier -- records calls into a per-model ledger
    without touching the CLI."""

    def __init__(self, model):
        self.model = model
        self.call_counts = Counter()
        self.prompts = []

    def complete(self, prompt, tier=None):
        self.call_counts[self.model] += 1
        self.prompts.append(prompt)
        return f"[{self.model}] reply"


def _tiered():
    tc = TieredCoder.__new__(TieredCoder)
    tc.analyst = _FakeCoder("haiku")
    tc.editor = _FakeCoder("sonnet")
    tc.architect = _FakeCoder("opus")
    return tc


def test_tier_routing():
    tc = _tiered()
    digest = tc.digest("EVIDENCE-BLOCK")
    _check("tier: digest goes to the Haiku analyst", "[haiku]" in digest)
    _check("tier: analyst prompt carries the analyst instruction",
           tc.analyst.prompts[0].startswith(_ANALYST[:20]))

    off = tc.complete("prompt", tier="editor")
    _check("tier: off-plateau edit goes to the Sonnet editor", "[sonnet]" in off)
    on = tc.complete("prompt", tier="architect")
    _check("tier: on-plateau edit goes to the Opus architect", "[opus]" in on)

    led = tc.call_counts
    _check("tier: cost ledger tallies each tier once",
           led["haiku"] == 1 and led["sonnet"] == 1 and led["opus"] == 1)
    _check("tier: routine work never touched the expensive architect tier "
           "until escalation", tc.architect.call_counts["opus"] == 1)


class _RecordingAgent:
    """Mutator-facing agent that records the tier it was asked for and returns a
    valid full-module candidate so Mutator yields a program."""

    def __init__(self):
        self.tiers = []
        self.call_counts = Counter()

    def digest(self, evidence):
        return evidence

    def complete(self, prompt, tier="editor"):
        self.tiers.append(tier)
        return "```python\ndef search(q, G):\n    return {1: 1.0}\n```"


def _failure():
    return [{"query": "q", "bucket": "missed",
             "metrics": {"recall@20": 0.3, "hit@1": 0.0, "hit@5": 0.0, "mrr": 0.1},
             "attribution": "GENERATION (gold not in top-100, ...)",
             "empty_result": False, "missed_gold": [], "top_wrong": [],
             "gold_ranks": [], "error": None}]


def test_mutator_plateau_selects_architect():
    from graphretr_opt.artifact.program import SearchProgram
    agent = _RecordingAgent()
    mut = Mutator(agent, "PRIMITIVES-DOC", SAFE_BUILTINS)
    prog = SearchProgram("def search(q, G):\n    return {0: 1.0}\n", family="t")

    cand, _, _ = mut.propose(prog, _failure(), [], [], edit_budget=4, plateau=False)
    _check("mutator: off-plateau asks the editor tier", agent.tiers[-1] == "editor")
    cand2, _, _ = mut.propose(prog, _failure(), [], [], edit_budget=4, plateau=True)
    _check("mutator: on-plateau asks the architect tier", agent.tiers[-1] == "architect")
    _check("mutator: produced a usable candidate from the tiered reply",
           cand is not None and cand2 is not None)


def main():
    test_tier_routing()
    test_mutator_plateau_selects_architect()
    print("\nall tiered tests passed")


if __name__ == "__main__":
    main()
