"""Unit tests for Phase 0.6a: the deterministic per-query cost meter (post-mortem
#5). The real cost driver is per-call rerank compute -- unmeasured in run-7 except
via noisy wall-clock latency_s. We count the items each program sends to
llm_rerank and thread it: RetrievalGraph._rerank_items -> RunStats.rerank_items ->
MetricVector.rerank_items (mean per query). Noise-free: a function of program +
query only, counted cache-or-not.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_cost_meter
No FalkorDB / no network needed (LLM + get_text stubbed).
"""
import os
import tempfile

from graphretr_opt.config import load_config
from starksearch.primitives import Allowlists
from starksearch.graph import RetrievalGraph
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


# The old "Sandbox.run threads the rerank_items delta" unit was removed with the
# in-process Sandbox (carve-out 2026-06-25): the subprocess worker
# (_worker_stark._one) now snapshots the SAME `_rerank_items` delta into its cost
# payload, and StarkRewardAdapter folds it into MetricVector.rerank_items. The
# counter itself (the real cost driver) is still unit-tested directly below.


# ---- 0.6a part 2: llm_rerank charges len(pool) items -------------------------

class _StubBudget:
    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user, model=None, max_tokens=400):
        self.calls += 1
        n = user.count("[")
        return {"scores": {str(i): 1.0 for i in range(n)}}


def _make_graph(tmpdir):
    g = RetrievalGraph.__new__(RetrievalGraph)
    g._cfg = load_config(root=tmpdir, rerank_pool_max=3)
    g._allow = Allowlists(labels=["drug"], rel_types=["ppi"], ntypes=["drug"])
    g._llm_budget = _StubBudget()
    g._llm_calls = 0
    g._pinned_query = None
    g._rerank_disk = os.path.join(tmpdir, "runs", "llm_rerank_cache.json")
    g._rerank_mem = None
    g.get_text = lambda ids, limit=50: {nid: f"node {nid}" for nid in ids}
    return g


def test_llm_rerank_charges_pool_size():
    with tempfile.TemporaryDirectory() as d:
        g = _make_graph(d)            # __init__ bypassed -> _rerank_items unset
        q = "which drug treats X"
        g._pin_query(q)
        g.llm_rerank(q, [9, 8, 7, 6, 5])   # pool_max=3 -> 3 items charged
        _check("llm_rerank charges the capped pool size (getattr-guarded init)",
               g._rerank_items == 3)
        g.llm_rerank(q, [9, 8, 7, 6, 5])   # cache hit, but the demand still counts
        _check("cost meter counts the rerank demand cache-or-not (deterministic)",
               g._rerank_items == 6)


# ---- 0.6a part 3: MetricVector exposes rerank_items (audit/MLflow) ------------

def test_metricvector_exposes_rerank_items():
    mv = MetricVector(rerank_items=12.5)
    _check("rerank_items is a top-level MetricVector field", mv.rerank_items == 12.5)
    _check("as_flat() emits rerank_items for MLflow/audit",
           mv.as_flat().get("rerank_items") == 12.5)


def main():
    test_llm_rerank_charges_pool_size()
    test_metricvector_exposes_rerank_items()
    print("\nall cost_meter tests passed")


if __name__ == "__main__":
    main()
