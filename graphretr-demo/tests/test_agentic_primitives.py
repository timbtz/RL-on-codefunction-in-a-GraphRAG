"""Unit tests for Item 2: the agentic primitives G.judge_sufficient and
G.pick_frontier, plus reformulate's new missing= steer. Covers query pinning,
the rerank_items cost meter, disk cache hit/restart, index->id mapping + top
cap on pick_frontier, and that `missing` forks reformulate's cache key while
missing=None stays byte-identical to the pre-Item-2 entries.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_agentic_primitives
No FalkorDB / no network / no API key needed (LLM + get_text are stubbed).
"""
import os
import tempfile

from graphretr_opt.config import load_config
from graphretr_opt.env.primitives import Allowlists
from graphretr_opt.env.retrieval_graph import RetrievalGraph


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


class _StubBudget:
    """Routes by system prompt: judge -> insufficient + 2 hints; frontier ->
    indices (incl. out-of-range + non-int to exercise validation); reformulate
    -> an echo of whether it saw a MISSING line."""

    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user, model=None, max_tokens=400):
        self.calls += 1
        if "sufficiency judge" in system:
            return {"sufficient": False, "missing": ["BRCA1 protein", "  ", 7]}
        if "EXPAND next" in system:
            return {"expand": [0, 2, 99, "x", 1, 2]}  # 99 OOR, "x" bad, 2 dup
        if "rewrite a biomedical" in system:
            return {"query": f"refined ({'MISSING' in user})"}
        return {}


def _make_graph(tmpdir, budget):
    g = RetrievalGraph.__new__(RetrievalGraph)
    g._cfg = load_config(root=tmpdir, judge_ctx_max=3, frontier_ctx_max=3,
                         reformulate_ctx_max=3)
    g._allow = Allowlists(labels=["drug"], rel_types=["ppi"], ntypes=["drug"])
    g._llm_budget = budget
    g._llm_calls = 0
    g._rerank_items = 0
    g._pinned_query = None
    g._judge_disk = os.path.join(tmpdir, "runs", "judge_cache.json")
    g._judge_mem = None
    g._frontier_disk = os.path.join(tmpdir, "runs", "frontier_cache.json")
    g._frontier_mem = None
    g._reformulate_disk = os.path.join(tmpdir, "runs", "reformulate_cache.json")
    g._reformulate_mem = None
    g.get_text = lambda ids, limit=50: {nid: f"node {nid}" for nid in ids}
    return g


def test_judge_pinning_cost_and_cache():
    with tempfile.TemporaryDirectory() as d:
        budget = _StubBudget()
        g = _make_graph(d, budget)
        try:
            g.judge_sufficient("q", [1, 2, 3])
            _check("judge: unpinned/off-query raises", False)
        except ValueError:
            _check("judge: unpinned/off-query raises", True)
        g._pin_query("which drug treats X")
        out = g.judge_sufficient("which drug treats X", [9, 8, 7, 6, 5], top=10)
        _check("judge: sufficient coerced to bool", out["sufficient"] is False)
        _check("judge: missing kept as <=N trimmed strings (junk dropped)",
               out["missing"] == ["BRCA1 protein"])
        _check("judge: context capped at judge_ctx_max",
               g._rerank_items == 3)
        _check("judge: metered as one llm call", g._llm_calls == 1)
        # repeat the same capped pool {9,8,7} reordered -> cache hit, no new call
        g.judge_sufficient("which drug treats X", [7, 8, 9])
        _check("judge: cache hit -> one billed call", budget.calls == 1)

        budget2 = _StubBudget()
        g2 = _make_graph(d, budget2)
        g2._pin_query("which drug treats X")
        g2.judge_sufficient("which drug treats X", [9, 8, 7])
        _check("judge: cache survives restart", budget2.calls == 0)


def test_frontier_maps_indices_and_top():
    with tempfile.TemporaryDirectory() as d:
        budget = _StubBudget()
        g = _make_graph(d, budget)
        g._pin_query("q")
        ids = [10, 20, 30, 40, 50]            # pool capped to first 3: 10,20,30
        out = g.pick_frontier("q", ids, top=2)
        _check("frontier: indices map back to valid node ids",
               all(n in ids for n in out))
        _check("frontier: out-of-range/non-int indices dropped, order kept",
               out == [10, 30])               # [:top] of resolved [10,30,20]
        _check("frontier: respects top", len(out) <= 2)
        _check("frontier: cost meter charged the sent pool", g._rerank_items == 3)
        _check("frontier: one billed call", g._llm_calls == 1)
        # cache hit (resolved ids cached) -- same pool reordered is free + correct
        out2 = g.pick_frontier("q", [30, 20, 10], top=8)
        _check("frontier: cache hit -> no new billed call", budget.calls == 1)
        _check("frontier: cached resolution is the same id set",
               set(out2) == {10, 30, 20})
        try:
            g.pick_frontier("other", ids)
            _check("frontier: off-query raises", False)
        except ValueError:
            _check("frontier: off-query raises", True)


def test_reformulate_missing_forks_cache_key():
    with tempfile.TemporaryDirectory() as d:
        budget = _StubBudget()
        g = _make_graph(d, budget)
        g._pin_query("q")
        a = g.reformulate([1, 2, 3])                       # no missing -> call 1
        _check("reformulate: missing=None did not see a MISSING line",
               a == "refined (False)")
        g.reformulate([3, 2, 1])                           # same key -> cache hit
        _check("reformulate: missing=None repeat is a cache hit", budget.calls == 1)
        b = g.reformulate([1, 2, 3], missing=["pathway"])  # forked key -> call 2
        _check("reformulate: missing forks the cache key", budget.calls == 2)
        _check("reformulate: missing steered the prompt (saw MISSING line)",
               b == "refined (True)")
        g.reformulate([1, 2, 3], missing=["pathway"])      # same missing -> hit
        _check("reformulate: same missing repeat is a cache hit", budget.calls == 2)


def main():
    test_judge_pinning_cost_and_cache()
    test_frontier_maps_indices_and_top()
    test_reformulate_missing_forks_cache_key()
    print("\nall agentic-primitive tests passed")


if __name__ == "__main__":
    main()
