"""Unit tests for the v7 RAW FLOOR -- the three trusted primitives the editable
seed is built on: G.query (read-only Cypher + cache + row cap), G.embed, and
G.llm (metered + disk-cached JSON call). Plus a gate check that the v7 seed is
sandbox-legal.

Read-only ENFORCEMENT is the backend's job (GRAPH.RO_QUERY) and is verified live
against FalkorDB, not here -- these are DB-free stub tests for behavior/caching.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_floor.py -q
"""
import os
import tempfile

from graphretr_opt.config import load_config
from graphretr_opt.env.primitives import Allowlists
from graphretr_opt.env.retrieval_graph import RetrievalGraph, _MAX_QUERY_ROWS
from graphretr_opt.env.cache import PrimitiveCache
from graphretr_opt.env.sandbox import check_source, compile_program


class _StubBudget:
    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user, model=None, max_tokens=400):
        self.calls += 1
        return {"echo": user[-20:], "model": model}


class _StubBackend:
    """Records ro_query calls; returns `rows` (ignores the query)."""
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
        self.query_count = 0

    def ro_query(self, query, params=None, timeout_ms=2000):
        self.calls += 1
        self.query_count += 1
        return [list(r) for r in self.rows]


def _graph(tmpdir, budget=None, backend=None, embedder=None):
    g = RetrievalGraph.__new__(RetrievalGraph)
    g._cfg = load_config(root=tmpdir)
    g._allow = Allowlists(labels=["drug"], rel_types=["ppi"], ntypes=["drug"])
    g._cache = PrimitiveCache()
    g._backend = backend
    g._embedder = embedder
    g._llm_budget = budget
    g._llm_calls = 0
    g._rerank_items = 0
    g._pinned_query = None
    g._llm_disk = os.path.join(tmpdir, "runs", "llm_cache.json")
    g._llm_mem = None
    g.get_text = lambda ids, limit=50: {nid: f"doc {nid}" for nid in ids}
    return g


def test_llm_returns_dict_meters_and_caches():
    with tempfile.TemporaryDirectory() as d:
        b = _StubBudget()
        g = _graph(d, budget=b)
        out1 = g.llm("system prompt", "the user question")
        assert isinstance(out1, dict) and "echo" in out1
        assert g._llm_calls == 1 and b.calls == 1
        # identical call -> served from cache, no new budget call
        out2 = g.llm("system prompt", "the user question")
        assert out2 == out1
        assert b.calls == 1, "second identical call must hit the cache"
        assert g._llm_calls == 2, "the call is still metered even on a cache hit"


def test_llm_appends_context_and_meters_items():
    with tempfile.TemporaryDirectory() as d:
        b = _StubBudget()
        g = _graph(d, budget=b)
        g.llm("sys", "q", context_ids=[5, 5, 9])  # dup 5 deduped
        # rerank_items cost meter charged the (deduped) context size
        assert g._rerank_items == 2


def test_llm_disabled_without_budget():
    with tempfile.TemporaryDirectory() as d:
        g = _graph(d, budget=None)
        try:
            g.llm("sys", "user")
            assert False, "llm() must raise when no budget/key"
        except ValueError:
            pass


def test_query_caches_and_caps_rows():
    with tempfile.TemporaryDirectory() as d:
        be = _StubBackend(rows=[[i, float(i)] for i in range(_MAX_QUERY_ROWS + 50)])
        g = _graph(d, backend=be)
        r1 = g.query("MATCH (n) RETURN n.id, n.score", {"x": 1})
        assert len(r1) == _MAX_QUERY_ROWS, "rows must be capped"
        assert isinstance(r1[0], list)
        # identical (cypher, params) -> cache hit, backend not re-queried
        r2 = g.query("MATCH (n) RETURN n.id, n.score", {"x": 1})
        assert r2 == r1 and be.calls == 1
        # different params -> new query
        g.query("MATCH (n) RETURN n.id, n.score", {"x": 2})
        assert be.calls == 2


def test_embed_returns_plain_list():
    class _E:
        def encode(self, text):
            class _V:
                def tolist(self_inner):
                    return [0.1, 0.2, 0.3]
            return _V()
    with tempfile.TemporaryDirectory() as d:
        g = _graph(d, embedder=_E())
        v = g.embed("query text")
        assert v == [0.1, 0.2, 0.3] and isinstance(v, list)


def test_v7_seed_is_sandbox_legal():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(
        root, "src/graphretr_opt/artifact/seeds/reasoning_first_v7.py")).read()
    check_source(src)        # raises SandboxError if illegal
    compile_program(src)     # builds the search() callable
