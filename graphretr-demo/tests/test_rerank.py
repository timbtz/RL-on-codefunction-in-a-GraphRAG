"""Unit tests for G.llm_rerank: pool cap, query pinning, budget ceiling, cache
hit/restart, determinism, and reasoning_first seed AST-gate acceptance.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_rerank
No FalkorDB / no network / no API key needed (LLM + get_text are stubbed).
"""
import os
import tempfile

from graphretr_opt.config import load_config
from graphretr_opt.env.openai_client import BudgetExceeded
from graphretr_opt.env.primitives import Allowlists
from graphretr_opt.env.retrieval_graph import RetrievalGraph
from graphretr_opt.env.sandbox import check_source


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


class _StubBudget:
    """Returns a descending score per candidate index; counts calls. raise_at
    makes chat_json raise BudgetExceeded to exercise the ceiling wall."""

    def __init__(self, raise_at=False):
        self.calls = 0
        self.raise_at = raise_at
        self.last_n = 0

    def chat_json(self, system, user, model=None, max_tokens=400):
        if self.raise_at:
            raise BudgetExceeded("ceiling hit")
        self.calls += 1
        n = user.count("[")  # one "[i]" marker per candidate line
        self.last_n = n
        return {"scores": {str(i): 1.0 - i / 100.0 for i in range(n)}}


def _make_graph(tmpdir, budget):
    g = RetrievalGraph.__new__(RetrievalGraph)
    g._cfg = load_config(root=tmpdir, rerank_pool_max=3)
    g._allow = Allowlists(labels=["drug"], rel_types=["ppi"], ntypes=["drug"])
    g._llm_budget = budget
    g._llm_calls = 0
    g._pinned_query = None
    g._rerank_disk = os.path.join(tmpdir, "runs", "llm_rerank_cache.json")
    g._rerank_mem = None
    # stub the DB-backed document fetch
    g.get_text = lambda ids, limit=50: {nid: f"node {nid}" for nid in ids}
    return g


def test_pinning_and_pool_cap():
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d, _StubBudget())
        q = "which drug treats X"
        try:
            g.llm_rerank(q, [1, 2, 3])
            _check("rerank: unpinned raises", False)
        except ValueError:
            _check("rerank: unpinned raises", True)
        g._pin_query(q)
        out = g.llm_rerank(q, [9, 8, 7, 6, 5], top=10)
        # pool_max=3 -> only the first 3 in CALLER order are scored
        _check("rerank: pool capped to rerank_pool_max", len(out) == 3)
        _check("rerank: returns sorted descending",
               [s for _, s in out] == sorted((s for _, s in out), reverse=True))
        _check("rerank: pool is the first-N in caller order",
               sorted(nid for nid, _ in out) == [7, 8, 9])
        _check("rerank: metered as one llm call", g._llm_calls == 1)


def test_budget_ceiling():
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d, _StubBudget(raise_at=True))
        q = "q"
        g._pin_query(q)
        try:
            g.llm_rerank(q, [1, 2])
            _check("rerank: budget ceiling propagates", False)
        except BudgetExceeded:
            _check("rerank: budget ceiling propagates", True)


def test_cache_and_restart():
    with tempfile.TemporaryDirectory() as d:
        budget = _StubBudget()
        g = _make_graph(d, budget)
        q = "cached query"
        g._pin_query(q)
        a = g.llm_rerank(q, [3, 1, 2])
        b = g.llm_rerank(q, [1, 2, 3])  # same pool (set), different order
        _check("rerank: cache hit -> one billed call", budget.calls == 1)
        _check("rerank: deterministic given cache", a == b)
        # fresh graph (new process) reuses the on-disk cache
        budget2 = _StubBudget()
        g2 = _make_graph(d, budget2)
        g2._pin_query(q)
        c = g2.llm_rerank(q, [1, 2, 3])
        _check("rerank: cache survives restart", budget2.calls == 0)
        _check("rerank: restart returns same order", c == a)


def test_seeds_pass_sandbox():
    root = load_config().root
    for name in ("reasoning_first", "reasoning_first_static"):
        path = os.path.join(root, "src/graphretr_opt/artifact/seeds", f"{name}.py")
        check_source(open(path).read())  # raises on violation
        _check(f"sandbox: seed {name} passes AST gate", True)


def main():
    test_pinning_and_pool_cap()
    test_budget_ceiling()
    test_cache_and_restart()
    test_seeds_pass_sandbox()
    print("\nall rerank tests passed")


if __name__ == "__main__":
    main()
