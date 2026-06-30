"""Offline tests for the STaRK FileSet seed + harness build path. No infra: the
FalkorDB backend, embedder, and LLM budget are monkeypatched with fakes that
return canned rows, so the seed is materialized, imported and run end-to-end in
this process.

Proves the carve-out's headline: the STaRK candidate is a whole FileSet over
`starksearch/src` whose materialized tree exposes StarkGraphSearchService, and
`build_service` wires the injected, metered clients so `service.search(q)`
returns a dict[int, float] -- the same surface graphsearch's FileSet path has.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "starksearch", "src"))

from graphretr_opt.artifact.file_set import FileSet  # noqa: E402
from graphretr_opt.config import load_config  # noqa: E402
import stark_harness.qa_runner as qa_runner  # noqa: E402


# A single dict that satisfies every prompt the service issues (extract reads
# keywords/anchor_ntypes/answer_ntype/rel_types; rerank reads scores;
# reformulate reads query) -- each helper picks only its own keys.
_LLM_REPLY = {
    "keywords": ["alpha", "beta"],
    "anchor_ntypes": ["disease"],
    "answer_ntype": "gene_protein",
    "rel_types": ["target"],
    "scores": {"0": 0.9, "1": 0.7, "2": 0.5},
    "query": "rewritten query",
}


class FakeBudget:
    def chat_json(self, system, user, model=None, max_tokens=400):
        return dict(_LLM_REPLY)


class FakeEmbedder:
    def encode(self, text):
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


class FakeBackend:
    def __init__(self, *a, **k):
        self.query_count = 0

    def configure(self, *a, **k):
        return True

    def ro_query(self, query, params=None, timeout_ms=2000):
        self.query_count += 1
        ql = query.lower()
        if "db.labels" in ql:
            return [["Entity"], ["disease"], ["gene_protein"]]
        if "relationshiptypes" in ql:
            return [["target"], ["carrier"]]
        if "distinct n.ntype" in ql:
            return [["disease"], ["gene_protein"]]
        if "db.idx.vector.querynodes" in ql:
            return [[1, 0.1], [2, 0.2], [3, 0.3]]          # (id, distance)
        if "db.idx.fulltext.querynodes" in ql:
            return [[1, 4.0], [2, 3.0]]
        if "return n.id, n.text" in ql:
            return [[1, "doc one"], [2, "doc two"], [3, "doc three"]]
        if "type(r)" in ql:
            return [[1, "target", 2], [2, "target", 3]]    # neighbors
        if "where n.ntype" in ql:
            return [[2], [3]]                               # filter_nodes
        if "algo.bfs" in ql:
            return [[3, "gene_protein"]]
        return []


def test_seed_materializes_and_imports():
    cfg = load_config()
    fs = FileSet.from_base(cfg.stark_src_abs, cfg.stark_editable_files)
    assert set(fs.overlay) == set(cfg.stark_editable_files)
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "src")
        fs.materialize(dest)
        assert os.path.exists(os.path.join(
            dest, "stark_search", "stark_graph_search_service.py"))
        assert os.path.exists(os.path.join(dest, "stark_harness", "qa_runner.py"))
        sys.path.insert(0, dest)
        try:
            mod = __import__(
                "stark_search.stark_graph_search_service",
                fromlist=["StarkGraphSearchService"])
            assert hasattr(mod, "StarkGraphSearchService")
        finally:
            sys.path.remove(dest)


def test_build_service_runs_offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(qa_runner, "FalkorDBBackend", FakeBackend)
    monkeypatch.setattr(qa_runner, "make_embedder", lambda cfg, budget: FakeEmbedder())
    monkeypatch.setattr(qa_runner, "OpenAIBudget", lambda *a, **k: FakeBudget())

    sink = qa_runner.CostSink()
    cfg = load_config()
    service = qa_runner.build_service(
        {"host": "x", "port": 0, "graph_name": "g"},
        {"root": cfg.root}, instrument=sink)
    assert type(service).__name__ == "StarkGraphSearchService"

    pred = qa_runner.run_query(service, "what gene targets this disease?")
    assert isinstance(pred, dict) and pred
    for k, v in pred.items():
        assert isinstance(k, int) and not isinstance(k, bool)
        assert isinstance(v, float)
    # the metered clients wrote into the shared sink
    assert sink.db_queries > 0 and sink.llm_calls > 0
