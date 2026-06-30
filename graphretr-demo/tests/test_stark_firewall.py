"""Unit tests for the STaRK harness FIREWALL -- the read-only DB proxy that is the
single chokepoint every candidate Cypher query passes through. No infra: a fake
backend records the query and returns canned rows.

The firewall is the ONLY thing standing between a mutated candidate's raw Cypher
and the SHARED FalkorDB container now that the candidate writes Cypher itself
(the old wall lived in RetrievalGraph methods the candidate routed around). It
must reject writes and runaway `algo.SPpaths(maxLen>=4)`, cap result rows, and
count db_queries ONLY on queries it actually lets through.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "starksearch", "src"))

from stark_harness.qa_runner import (  # noqa: E402
    CostSink, ReadOnlyGraphClient, _MAX_QUERY_ROWS, _assert_safe)
from graphretr_opt.config import load_config  # noqa: E402


class FakeBackend:
    """Records the last (query, params) and returns canned rows. Answers the
    allowlist-bootstrap queries ReadOnlyGraphClient issues at construction."""
    def __init__(self, rows=None):
        self._rows = rows if rows is not None else [[1, 0.0]]
        self.calls = []

    def configure(self, *a, **k):
        return True

    def ro_query(self, query, params=None, timeout_ms=2000):
        self.calls.append((query, params))
        ql = query.lower()
        if "db.labels" in ql:
            return [["Entity"], ["disease"], ["gene_protein"]]
        if "relationshiptypes" in ql:
            return [["carrier"], ["target"]]
        if "distinct n.ntype" in ql:
            return [["disease"], ["gene_protein"]]
        return self._rows


def _client(rows=None):
    return ReadOnlyGraphClient(FakeBackend(rows), CostSink(), load_config())


def test_allowlists_bootstrapped():
    db = _client()
    assert "Entity" not in db.labels and "disease" in db.labels
    assert set(db.rel_types) == {"carrier", "target"}
    assert set(db.ntypes) == {"disease", "gene_protein"}


def test_assert_safe_accepts_read_only():
    for q in (
        "MATCH (n:Entity) RETURN n.id LIMIT 5",
        "CALL db.idx.vector.queryNodes($l,'embedding',$k,vecf32($v)) "
        "YIELD node, score RETURN node.id, score",
        "UNWIND $ids AS i MATCH (s:Entity {id:i}) CALL algo.bfs(s, 2, null) "
        "YIELD nodes UNWIND nodes AS n RETURN DISTINCT n.id LIMIT 100",
        "CALL algo.SPpaths({maxLen:3, pathCount:1}) YIELD path RETURN path",
    ):
        assert _assert_safe(q) == q


@pytest.mark.parametrize("q", [
    "CREATE (n:Entity {id:1})",
    "create (n:Entity {id:1})",
    "MATCH (n) SET n.x = 1",
    "MATCH (n) DETACH DELETE n",
    "MERGE (n:Entity {id:1})",
    "MATCH (n) REMOVE n.x",
    "MATCH (n) DROP CONSTRAINT foo",
    "CALL db.create.setNodeVectorProperty(n, 'embedding', $v)",
    "MATCH (n) // delete\n SET n.y = 2",  # comment doesn't hide the write
])
def test_assert_safe_rejects_writes(q):
    with pytest.raises(ValueError):
        _assert_safe(q)


@pytest.mark.parametrize("q", [
    "CALL algo.SPpaths({sourceNode:a, targetNode:b, maxLen:4}) YIELD path RETURN path",
    "CALL algo.SPpaths({maxLen:5, pathCount:1}) YIELD path RETURN path",
    "CALL algo.SPpaths({sourceNode:a, targetNode:b}) YIELD path RETURN path",  # no maxLen
])
def test_assert_safe_rejects_runaway_sppaths(q):
    with pytest.raises(ValueError):
        _assert_safe(q)


def test_cypher_counts_only_allowed_queries():
    sink = CostSink()
    db = ReadOnlyGraphClient(FakeBackend(), sink, load_config())
    sink.reset()                      # ignore bootstrap reads
    db.cypher("MATCH (n:Entity) RETURN n.id LIMIT 3")
    assert sink.db_queries == 1
    with pytest.raises(ValueError):
        db.cypher("MATCH (n) DETACH DELETE n")
    assert sink.db_queries == 1       # rejected query never reached the backend
    with pytest.raises(ValueError):
        db.cypher("CALL algo.SPpaths({maxLen:9}) YIELD path RETURN path")
    assert sink.db_queries == 1


def test_cypher_truncates_rows_to_cap():
    big = [[i, 0.0] for i in range(_MAX_QUERY_ROWS + 250)]
    db = _client(rows=big)
    out = db.cypher("MATCH (n:Entity) RETURN n.id, n.x LIMIT 999999")
    assert len(out) == _MAX_QUERY_ROWS
    assert isinstance(out[0], list)
