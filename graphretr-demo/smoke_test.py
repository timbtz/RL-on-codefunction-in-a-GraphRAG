#!/usr/bin/env python
"""
smoke_test.py -- Stage 1a end-to-end verification (PASS/FAIL per check).

Asserts:
  1. STaRK prime loads in Python (SKB + QA).
  2. FalkorDB graph 'prime' holds the expected node/edge counts.
  3. An ad-hoc Cypher vector (ANN) search returns node-ids.
  4. An ad-hoc 1-hop neighbor query returns node-ids.
  5. The MLflow tracking server /health endpoint responds.

These queries are deliberately ad-hoc verification only -- NOT the reusable
retrieval-primitive library (that is the next stage).
"""
import os
import re
import sys
import json
import urllib.request

HOST = os.environ.get("FALKOR_HOST", "127.0.0.1")
PORT = int(os.environ.get("FALKOR_PORT", "6380"))
GRAPH_NAME = os.environ.get("GRAPH_NAME", "prime")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        results.append((name, False, repr(e)))
        print(f"[FAIL] {name}: {e!r}")


def sanitize(name):
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return ("_" + s) if s and s[0].isdigit() else s


# Shared handles populated by the checks.
state = {}


def c1_stark_loads():
    from stark_qa import load_qa, load_skb
    skb = load_skb("prime", download_processed=True)
    qa = load_qa("prime")
    state["skb"] = skb
    state["qa"] = qa
    return f"SKB {skb.num_nodes()} nodes / {skb.num_edges()} edges; QA {len(qa)} queries"


def c2_falkordb_counts():
    from falkordb import FalkorDB
    skb = state["skb"]
    g = FalkorDB(host=HOST, port=PORT).select_graph(GRAPH_NAME)
    state["g"] = g
    n = int(g.query("MATCH (n) RETURN count(n)").result_set[0][0])
    e = int(g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0])
    exp_n, exp_e = skb.num_nodes(), skb.num_edges()
    assert n == exp_n, f"node count {n} != expected {exp_n}"
    assert e == exp_e, f"edge count {e} != expected {exp_e}"
    return f"nodes={n} edges={e} (match expected)"


def c3_vector_search():
    from sentence_transformers import SentenceTransformer
    g = state["g"]
    skb = state["skb"]
    # Embed a real query so the ANN search is meaningful.
    qa = state["qa"]
    query_text, _, _, _ = qa[int(qa.get_idx_split()["test"][0])]
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    vec = enc.encode([query_text], normalize_embeddings=True)[0].tolist()
    label = sanitize("gene/protein")  # gene_protein -- the largest answerable type
    res = g.query(
        f"CALL db.idx.vector.queryNodes('{label}','embedding',10,vecf32($v)) "
        "YIELD node, score RETURN node.id AS id, score ORDER BY score",
        params={"v": vec},
    ).result_set
    ids = [int(r[0]) for r in res]
    assert len(ids) > 0, "vector search returned no nodes"
    assert all(isinstance(i, int) for i in ids)
    return f"ANN over :{label} returned {len(ids)} node-ids, top={ids[:5]}"


def c4_one_hop():
    g = state["g"]
    # Pick a node that actually has outgoing edges.
    seed = g.query(
        "MATCH (a:Entity)-->() RETURN a.id LIMIT 1"
    ).result_set[0][0]
    res = g.query(
        "MATCH (a:Entity {id:$id})-->(b) RETURN b.id AS id LIMIT 25",
        params={"id": int(seed)},
    ).result_set
    ids = [int(r[0]) for r in res]
    assert len(ids) > 0, f"node {seed} returned no 1-hop neighbors"
    return f"node {seed} -> {len(ids)} neighbor node-ids, sample={ids[:5]}"


def c5_mlflow_health():
    with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=5) as r:
        body = r.read().decode().strip()
        assert r.status == 200, f"status {r.status}"
    return f"{MLFLOW_URL}/health -> 200 ({body})"


if __name__ == "__main__":
    check("1. STaRK prime loads", c1_stark_loads)
    check("2. FalkorDB prime node/edge counts", c2_falkordb_counts)
    check("3. ad-hoc vector (ANN) search returns node-ids", c3_vector_search)
    check("4. ad-hoc 1-hop neighbor query returns node-ids", c4_one_hop)
    check("5. MLflow /health responds", c5_mlflow_health)

    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== {npass}/{len(results)} checks PASSED ====")
    sys.exit(0 if npass == len(results) else 1)
