"""Unit tests for Phase C1 G.reformulate: query pinning, context-pool cap,
budget ceiling, cache hit/restart, determinism, non-empty-string contract and
the empty-model fallback to the original query; plus reasoning_first_v6 seed
AST-gate acceptance.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_reformulate
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
    """Returns a canned rewritten query echoing how many context docs it saw.
    raise_at exercises the ceiling wall; empty returns no query (fallback path)."""

    def __init__(self, raise_at=False, empty=False):
        self.calls = 0
        self.raise_at = raise_at
        self.empty = empty
        self.last_ctx = 0

    def chat_json(self, system, user, model=None, max_tokens=400):
        if self.raise_at:
            raise BudgetExceeded("ceiling hit")
        self.calls += 1
        self.last_ctx = user.count("\n- ")
        if self.empty:
            return {"query": "   "}
        return {"query": f"refined query with {self.last_ctx} ctx"}


def _make_graph(tmpdir, budget):
    g = RetrievalGraph.__new__(RetrievalGraph)
    g._cfg = load_config(root=tmpdir, reformulate_ctx_max=3)
    g._allow = Allowlists(labels=["drug"], rel_types=["ppi"], ntypes=["drug"])
    g._llm_budget = budget
    g._llm_calls = 0
    g._pinned_query = None
    g._reformulate_disk = os.path.join(tmpdir, "runs", "reformulate_cache.json")
    g._reformulate_mem = None
    g.get_text = lambda ids, limit=50: {nid: f"node {nid}" for nid in ids}
    return g


def test_pinning_and_ctx_cap():
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d, _StubBudget())
        try:
            g.reformulate([1, 2, 3])
            _check("reformulate: unpinned raises", False)
        except ValueError:
            _check("reformulate: unpinned raises", True)
        g._pin_query("which drug treats X")
        out = g.reformulate([9, 8, 7, 6, 5], top=10)
        _check("reformulate: returns a non-empty str",
               isinstance(out, str) and out.strip())
        _check("reformulate: context capped at reformulate_ctx_max",
               g._llm_budget.last_ctx == 3)
        _check("reformulate: metered as one llm call", g._llm_calls == 1)


def test_budget_ceiling():
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d, _StubBudget(raise_at=True))
        g._pin_query("q")
        try:
            g.reformulate([1, 2])
            _check("reformulate: budget ceiling propagates", False)
        except BudgetExceeded:
            _check("reformulate: budget ceiling propagates", True)


def test_cache_and_restart():
    with tempfile.TemporaryDirectory() as d:
        budget = _StubBudget()
        g = _make_graph(d, budget)
        g._pin_query("cached query")
        a = g.reformulate([3, 1, 2])
        b = g.reformulate([1, 2, 3])  # same pool (set), different order
        _check("reformulate: cache hit -> one billed call", budget.calls == 1)
        _check("reformulate: deterministic given cache", a == b)
        budget2 = _StubBudget()
        g2 = _make_graph(d, budget2)
        g2._pin_query("cached query")
        c = g2.reformulate([1, 2, 3])
        _check("reformulate: cache survives restart", budget2.calls == 0)
        _check("reformulate: restart returns same query", c == a)


def test_empty_fallback():
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d, _StubBudget(empty=True))
        g._pin_query("original question")
        out = g.reformulate([1, 2])
        _check("reformulate: empty model output falls back to original q",
               out == "original question")


def test_seed_v6_passes_sandbox():
    root = load_config().root
    path = os.path.join(root, "src/graphretr_opt/artifact/seeds",
                        "reasoning_first_v6.py")
    check_source(open(path).read())  # raises on violation
    _check("sandbox: seed reasoning_first_v6 passes AST gate", True)


def main():
    test_pinning_and_ctx_cap()
    test_budget_ceiling()
    test_cache_and_restart()
    test_empty_fallback()
    test_seed_v6_passes_sandbox()
    print("\nall reformulate tests passed")


if __name__ == "__main__":
    main()
