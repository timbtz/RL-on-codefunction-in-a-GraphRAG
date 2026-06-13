"""Unit tests for the Phase-2 additions: OpenAIBudget ceiling math, the gate
complexity cap, extract() validation/pinning/caching (stubbed LLM -- no
network), sandbox llm_calls metering, and seed AST-gate acceptance.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_phase2
No FalkorDB / no network / no API key needed.
"""
import json
import os
import tempfile

from graphretr_opt.config import load_config
from graphretr_opt.env.openai_client import OpenAIBudget, BudgetExceeded
from graphretr_opt.env.sandbox import check_source
from graphretr_opt.optimizer.gate import Gate
from graphretr_opt.reward.objectives import code_complexity


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


class _MV:
    def __init__(self, recall, mrr, complexity=0.0, crashed=False):
        self._q = {"recall@20": recall, "mrr": mrr}
        self.code_complexity = complexity
        self.crashed = crashed

    def get(self, k):
        return self._q[k]


def test_budget():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "usage.json")
        b = OpenAIBudget(path, ceiling_usd=1.0)
        # spend tracking math: 1M in + 1M out of gpt-4o-mini = 0.15 + 0.60
        b._record("gpt-4o-mini", 1_000_000, 1_000_000)
        _check("budget: $ math", abs(b.spent_usd - 0.75) < 1e-9)
        _check("budget: ledger persisted",
               json.load(open(path))["requests"] == 1)
        # second instance resumes the ledger
        b2 = OpenAIBudget(path, ceiling_usd=1.0)
        _check("budget: ledger resumed", abs(b2.spent_usd - 0.75) < 1e-9)
        # ceiling is a wall BEFORE the call
        b2._record("gpt-4o-mini", 1_000_000, 1_000_000)
        try:
            b2._check()
            _check("budget: ceiling raises", False)
        except BudgetExceeded:
            _check("budget: ceiling raises", True)


def test_gate_complexity_cap():
    inc = _MV(0.40, 0.20)
    g = Gate(mode="blend", max_complexity=900)
    _check("cap: better-but-bloated rejects",
           not g.accept(_MV(0.50, 0.30, complexity=2456), inc))
    _check("cap: better-and-lean accepts",
           g.accept(_MV(0.50, 0.30, complexity=254), inc))
    g_off = Gate(mode="blend", max_complexity=0)
    _check("cap: 0 disables the wall",
           g_off.accept(_MV(0.50, 0.30, complexity=99999), inc))


class _StubBudget:
    """Returns a canned extract payload; counts calls."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_json(self, system, user, model=None, max_tokens=400):
        self.calls += 1
        return dict(self.payload)


def _make_graph(tmpdir, payload):
    """RetrievalGraph without a DB: bypass __init__, set only what extract needs."""
    from graphretr_opt.env.retrieval_graph import RetrievalGraph
    from graphretr_opt.env.primitives import Allowlists
    g = RetrievalGraph.__new__(RetrievalGraph)
    g._cfg = load_config(root=tmpdir)
    g._allow = Allowlists(
        labels=["disease", "drug", "gene_protein"],
        rel_types=["indication", "ppi"],
        ntypes=["disease", "drug", "gene/protein"])
    g._llm_budget = _StubBudget(payload)
    g._llm_calls = 0
    g._pinned_query = None
    g._extract_disk = os.path.join(tmpdir, "runs", "extract_cache.json")
    g._extract_mem = None
    return g


def test_extract():
    payload = {"keywords": ["aspirin", 42, "  headache  "],
               "anchor_ntypes": ["drug", "bogus_type"],
               "answer_ntype": "gene/protein",
               "rel_types": ["ppi", "not_a_rel"]}
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d, payload)
        q = "which gene does aspirin target"

        # pinning: only the current query is accepted
        try:
            g.extract(q)
            _check("extract: unpinned raises", False)
        except ValueError:
            _check("extract: unpinned raises", True)
        g._pin_query(q)
        try:
            g.extract("some node text the program looped over")
            _check("extract: non-query text raises", False)
        except ValueError:
            _check("extract: non-query text raises", True)

        out = g.extract(q)
        _check("extract: keywords cleaned",
               out["keywords"] == ["aspirin", "headache"])
        _check("extract: bogus ntype dropped", out["anchor_ntypes"] == ["drug"])
        _check("extract: answer_ntype kept", out["answer_ntype"] == "gene/protein")
        _check("extract: answer_label derived", out["answer_label"] == "gene_protein")
        _check("extract: bogus rel dropped", out["rel_types"] == ["ppi"])
        _check("extract: invocation metered", g._llm_calls == 1)

        # cache: second call costs nothing, still counts as an invocation
        out2 = g.extract(q)
        _check("extract: disk-cached (no second LLM call)",
               g._llm_budget.calls == 1)
        _check("extract: cached call still metered", g._llm_calls == 2)
        out2["keywords"].append("mutated")
        _check("extract: cache hands out copies",
               g.extract(q)["keywords"] == ["aspirin", "headache"])

        # a fresh graph (new process) reuses the disk cache
        g2 = _make_graph(d, payload)
        g2._pin_query(q)
        g2.extract(q)
        _check("extract: cache survives restart", g2._llm_budget.calls == 0)


def test_seeds_pass_sandbox():
    root = load_config().root
    for name in ("vector_only", "hybrid_rrf", "extract_first", "run1_best"):
        path = os.path.join(root, "src/graphretr_opt/artifact/seeds", f"{name}.py")
        src = open(path).read()
        check_source(src)  # raises on violation
        _check(f"sandbox: seed {name} passes AST gate "
               f"(complexity {code_complexity(src):.0f})", True)


def main():
    test_budget()
    test_gate_complexity_cap()
    test_extract()
    test_seeds_pass_sandbox()
    print("\nall phase2 tests passed")


if __name__ == "__main__":
    main()
