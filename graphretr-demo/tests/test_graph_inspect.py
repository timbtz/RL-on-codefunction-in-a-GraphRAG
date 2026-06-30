"""Unit tests for the read-only graph inspection gate + per-miss attribution
(optimizer/graph_inspect.py). Mirrors the STaRK firewall test (test_stark_firewall).

Pure-logic: a programmable FakeGraph stands in for the injected Neo4jGraph, so no
Neo4j / network is needed. Covers (1) the write firewall, (2) forced-LIMIT +
truncation, (3) the NOT_INGESTED / ORPHANED / UNREACHABLE / RANKING buckets.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_graph_inspect
"""
from graphretr_opt.optimizer.graph_inspect import (
    assert_read_only, GraphInspector,
)


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


class FakeGraph:
    """Programmable stand-in for langchain_neo4j.Neo4jGraph.

    `present` = node ids that exist; `adj` = {id: [neighbor {id,labels} ...]}.
    Returns canned rows by sniffing the (already firewalled + LIMIT-forced) cypher,
    and records every query so tests can assert the LIMIT was appended.
    """

    def __init__(self, present=(), adj=None, rowfactory=None):
        self.present = set(present)
        self.adj = dict(adj or {})
        self.calls = []                 # list of (cypher, params)
        self._rowfactory = rowfactory   # optional override for arbitrary inspect()

    def query(self, cypher, params=None):
        params = params or {}
        self.calls.append((cypher, params))
        c = cypher.lower()
        if self._rowfactory is not None:
            r = self._rowfactory(cypher, params)
            if r is not None:
                return r
        # find_node: MATCH (n {id:$id}) RETURN n.id, labels(n)
        if "labels(n) as labels" in c and "{id: $id}" in c:
            nid = params.get("id")
            return [{"id": nid, "labels": ["Document"]}] if nid in self.present else []
        # _has_path_from_seeds: MATCH p = (s)-[*..]-(g) RETURN count(p) AS c
        if "match p =" in c and "count(p) as c" in c:
            gold = params.get("gold")
            seeds = params.get("seeds") or []
            reachable = any(gold in [n["id"] for n in self.adj.get(s, [])] for s in seeds)
            return [{"c": 1 if reachable else 0}]
        # neighbors: [*1..d]-(m) RETURN DISTINCT m.id AS id
        if "distinct m.id as id" in c:
            return list(self.adj.get(params.get("id"), []))
        return []


def test_firewall_rejects_writes():
    for bad in ["CREATE (x)", "MATCH (n) SET n.x=1", "MATCH (n) DETACH DELETE n",
                "MERGE (a)-[:R]->(b)", "MATCH (n) REMOVE n.p", "DROP CONSTRAINT c",
                "CALL apoc.create.node(['L'],{})", "CALL db.create.setNodeVectorProperty(n)"]:
        try:
            assert_read_only(bad)
            _check(f"rejects: {bad[:24]}", False)
        except ValueError:
            _check(f"rejects: {bad[:24]}", True)


def test_firewall_allows_reads_and_ignores_comments():
    ok = "MATCH (n:Document) RETURN n.id LIMIT 5"
    _check("allows MATCH/RETURN", assert_read_only(ok) == ok)
    # a write keyword hidden in a comment must NOT trip the gate...
    _check("comment write ignored",
           assert_read_only("MATCH (n) RETURN n // todo: delete later") is not None)
    # ...but a real write still trips even with comments around it.
    try:
        assert_read_only("/* read */ MATCH (n) DELETE n")
        _check("real write past comment rejected", False)
    except ValueError:
        _check("real write past comment rejected", True)


def test_inspect_forces_limit_and_truncates():
    # rowfactory returns 100 rows for a no-LIMIT query; inspector caps to max_rows.
    g = FakeGraph(rowfactory=lambda c, p: [{"i": i} for i in range(100)]
                  if "return n.x" in c.lower() else None)
    gi = GraphInspector(g, db="graph_test", max_rows=10)
    rows = gi.inspect("MATCH (n) RETURN n.x")
    _check("truncated to max_rows", len(rows) == 10)
    _check("LIMIT appended when absent", "limit 10" in g.calls[-1][0].lower())
    # a query that already has a LIMIT is not given a second one.
    gi.inspect("MATCH (n) RETURN n.x LIMIT 3")
    appended = g.calls[-1][0].lower().count("limit")
    _check("no double LIMIT", appended == 1)


def test_attribution_not_ingested():
    gi = GraphInspector(FakeGraph(present=set()), db="g")
    msg = gi.attribute_miss("doc:absent")
    _check("NOT_INGESTED when gold absent", msg.startswith("NOT_INGESTED"))


def test_attribution_ranking_beats_graph_probe():
    # gold present AND in retrieved -> RANKING (search fix), regardless of graph.
    gi = GraphInspector(FakeGraph(present={"doc:x"}), db="g")
    msg = gi.attribute_miss("doc:x", retrieved_ids=["doc:x", "doc:y"])
    _check("RANKING when gold retrieved", msg.startswith("RANKING"))


def test_attribution_orphaned_no_neighbours():
    # present, not retrieved, degree-0 -> ORPHANED (ingestion fix).
    gi = GraphInspector(FakeGraph(present={"doc:x"}, adj={}), db="g")
    msg = gi.attribute_miss("doc:x", retrieved_ids=["doc:other"])
    _check("ORPHANED when no neighbours", msg.startswith("ORPHANED"))


def test_attribution_unreachable_connected_not_retrieved():
    # present, has neighbours, not retrieved -> UNREACHABLE (search fix).
    g = FakeGraph(present={"doc:x"}, adj={"doc:x": [{"id": "person:tim", "labels": ["Person"]}]})
    gi = GraphInspector(g, db="g")
    msg = gi.attribute_miss("doc:x", retrieved_ids=["doc:other"])
    _check("UNREACHABLE when connected but not reached", msg.startswith("UNREACHABLE"))


def test_attribution_orphaned_with_seeds_no_path():
    # seeds supplied, no bounded path from any seed -> ORPHANED.
    g = FakeGraph(present={"doc:x"}, adj={"chunk:s": [{"id": "doc:other", "labels": ["Document"]}]})
    gi = GraphInspector(g, db="g")
    msg = gi.attribute_miss("doc:x", seed_ids=["chunk:s"], retrieved_ids=[])
    _check("ORPHANED when no path from seeds", msg.startswith("ORPHANED"))


def test_attribution_unreachable_with_seeds_path_exists():
    # seeds supplied, a bounded path from a seed reaches gold, not retrieved -> UNREACHABLE.
    g = FakeGraph(present={"doc:x"}, adj={"chunk:s": [{"id": "doc:x", "labels": ["Document"]}]})
    gi = GraphInspector(g, db="g")
    msg = gi.attribute_miss("doc:x", seed_ids=["chunk:s"], retrieved_ids=[])
    _check("UNREACHABLE when seed-path exists but search missed", msg.startswith("UNREACHABLE"))


def main():
    test_firewall_rejects_writes()
    test_firewall_allows_reads_and_ignores_comments()
    test_inspect_forces_limit_and_truncates()
    test_attribution_not_ingested()
    test_attribution_ranking_beats_graph_probe()
    test_attribution_orphaned_no_neighbours()
    test_attribution_unreachable_connected_not_retrieved()
    test_attribution_orphaned_with_seeds_no_path()
    test_attribution_unreachable_with_seeds_path_exists()
    print("\nall graph_inspect tests passed")


if __name__ == "__main__":
    main()
