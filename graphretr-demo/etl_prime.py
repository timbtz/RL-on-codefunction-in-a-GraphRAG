#!/usr/bin/env python
"""
etl_prime.py -- one-time ETL of STaRK 'prime' into FalkorDB.

Idempotent + persisted:
  * Node text embeddings (all-MiniLM-L6-v2, 384-d, CPU) are cached to
    emb_cache.npy so a re-run NEVER re-embeds.
  * If the FalkorDB graph already holds the full node+edge counts, the script
    exits immediately ("already loaded").
  * Otherwise it rebuilds the graph from scratch (delete + reload) using the
    cached embeddings, so a partial/aborted load self-heals.

Graph layout (see README):
  * Every node gets a shared label :Entity (carrying a range index on `id`
    for O(log n) edge MATCH) PLUS a per-type label (sanitized STaRK node-type,
    e.g. gene/protein -> gene_protein). Properties: {id, ntype, text, embedding}.
  * Edges are typed by sanitized STaRK relation name (e.g. off-label use ->
    off_label_use), directed src(edge_index[0]) -> dst(edge_index[1]).
  * A cosine vector index (dim 384) on `embedding` per per-type label.
"""
import os

# Use all CPU cores for the (CPU-only) embedding pass. Must be set before
# numpy / torch import so OpenMP/MKL pick it up.
_NTHREADS = str(os.cpu_count() or 4)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _NTHREADS)

import re
import json
import time
import numpy as np

HOST = os.environ.get("FALKOR_HOST", "127.0.0.1")
PORT = int(os.environ.get("FALKOR_PORT", "6380"))
GRAPH_NAME = os.environ.get("GRAPH_NAME", "prime")
MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384

EMB_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emb_cache.npy")
EMB_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emb_cache.meta.json")

NODE_BATCH = 1000      # nodes per UNWIND CREATE
EDGE_BATCH = 20000     # edges per UNWIND MATCH+CREATE
EMBED_BATCH = 256      # docs per encoder forward pass
MAX_SEQ_LEN = 128      # cap tokens/doc: ~2x faster CPU embed, salient fields are first


def sanitize(name: str) -> str:
    """STaRK type/relation -> safe Cypher label/reltype (alnum + underscore)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    if s and s[0].isdigit():
        s = "_" + s
    return s


def graph_counts(g):
    try:
        n = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
    except Exception:
        n = 0
    try:
        e = g.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
    except Exception:
        e = 0
    return int(n), int(e)


def compute_or_load_embeddings(skb, n_nodes):
    """Return float32 array [n_nodes, DIM]; persist to cache so we embed once."""
    if os.path.exists(EMB_CACHE) and os.path.exists(EMB_META):
        meta = json.load(open(EMB_META))
        if meta.get("n") == n_nodes and meta.get("model") == MODEL_NAME and meta.get("dim") == DIM:
            emb = np.load(EMB_CACHE)
            if emb.shape == (n_nodes, DIM):
                print(f"[embed] using cached embeddings {emb.shape} from {EMB_CACHE}")
                return emb
        print("[embed] cache present but stale -> recomputing")

    import torch
    torch.set_num_threads(int(_NTHREADS))
    from sentence_transformers import SentenceTransformer
    print(f"[embed] torch threads={torch.get_num_threads()}")
    print(f"[embed] building {n_nodes} node documents via skb.get_doc_info ...")
    t0 = time.time()
    docs = [skb.get_doc_info(i) for i in range(n_nodes)]
    print(f"[embed] docs built in {time.time()-t0:.1f}s; loading {MODEL_NAME} (CPU) ...")
    enc = SentenceTransformer(MODEL_NAME, device="cpu")
    enc.max_seq_length = MAX_SEQ_LEN
    print(f"[embed] max_seq_length={enc.max_seq_length}")
    t0 = time.time()
    emb = enc.encode(
        docs,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"[embed] encoded {emb.shape} in {time.time()-t0:.1f}s")
    np.save(EMB_CACHE, emb)
    json.dump({"n": n_nodes, "model": MODEL_NAME, "dim": DIM}, open(EMB_META, "w"))
    print(f"[embed] cached -> {EMB_CACHE}")
    return emb


def load_nodes(g, skb, emb, label_map):
    n_nodes = skb.num_nodes()
    node_type_ids = skb.node_types.numpy()
    print(f"[nodes] loading {n_nodes} nodes grouped by label ...")
    t0 = time.time()
    total = 0
    for tid, tname in skb.node_type_dict.items():
        label = label_map[tname]
        ids = np.where(node_type_ids == tid)[0]
        for s in range(0, len(ids), NODE_BATCH):
            chunk = ids[s:s + NODE_BATCH]
            rows = [{
                "id": int(i),
                "ntype": tname,
                "text": skb.get_doc_info(int(i)),
                "emb": emb[i].tolist(),
            } for i in chunk]
            g.query(
                f"UNWIND $rows AS row CREATE (n:Entity:`{label}` "
                f"{{id:row.id, ntype:row.ntype, text:row.text, embedding:vecf32(row.emb)}})",
                params={"rows": rows},
            )
            total += len(chunk)
        print(f"  {tname:24s} -> :{label:22s} {len(ids):7d} nodes  (cumulative {total})")
    print(f"[nodes] {total} nodes loaded in {time.time()-t0:.1f}s")


def create_indexes(g, label_map):
    print("[index] range index on :Entity(id) ...")
    g.query("CREATE INDEX FOR (n:Entity) ON (n.id)")
    print("[index] vector indexes (cosine, dim 384) per label ...")
    for label in sorted(set(label_map.values())):
        g.query(
            f"CREATE VECTOR INDEX FOR (n:`{label}`) ON (n.embedding) "
            f"OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}"
        )
        print(f"  vector index on :{label}(embedding)")


def load_edges(g, skb, rel_map):
    src = skb.edge_index[0].numpy()
    dst = skb.edge_index[1].numpy()
    etypes = skb.edge_types.numpy()
    n_edges = len(etypes)
    print(f"[edges] loading {n_edges} edges grouped by relation type ...")
    t0 = time.time()
    total = 0
    for eid, ename in skb.edge_type_dict.items():
        rel = rel_map[ename]
        mask = np.where(etypes == eid)[0]
        s_arr = src[mask]
        d_arr = dst[mask]
        cnt = len(mask)
        for s in range(0, cnt, EDGE_BATCH):
            rows = [{"s": int(a), "d": int(b)}
                    for a, b in zip(s_arr[s:s + EDGE_BATCH], d_arr[s:s + EDGE_BATCH])]
            g.query(
                "UNWIND $rows AS row MATCH (a:Entity {id:row.s}), (b:Entity {id:row.d}) "
                f"CREATE (a)-[:`{rel}`]->(b)",
                params={"rows": rows},
            )
            total += len(rows)
        print(f"  {ename:24s} -> [:{rel:22s}] {cnt:8d} edges  (cumulative {total}, {time.time()-t0:.0f}s)")
    print(f"[edges] {total} edges loaded in {time.time()-t0:.1f}s")


def main():
    from stark_qa import load_skb
    from falkordb import FalkorDB

    t_all = time.time()
    print("Loading STaRK 'prime' SKB ...")
    skb = load_skb("prime", download_processed=True)
    exp_nodes, exp_edges = skb.num_nodes(), skb.num_edges()
    print(f"expected: {exp_nodes} nodes / {exp_edges} edges")

    label_map = {t: sanitize(t) for t in skb.node_type_dict.values()}
    rel_map = {r: sanitize(r) for r in skb.edge_type_dict.values()}

    db = FalkorDB(host=HOST, port=PORT)
    g = db.select_graph(GRAPH_NAME)
    cur_n, cur_e = graph_counts(g)
    print(f"current graph '{GRAPH_NAME}': {cur_n} nodes / {cur_e} edges")
    if cur_n == exp_nodes and cur_e == exp_edges:
        print("[skip] graph already fully loaded -> nothing to do.")
        return

    # Embeddings first (the expensive, cached step) -- needed even if we rebuild.
    emb = compute_or_load_embeddings(skb, exp_nodes)

    # Rebuild from scratch for a clean, correct load.
    if cur_n or cur_e:
        print(f"[reset] graph partially loaded ({cur_n}/{cur_e}); deleting and rebuilding ...")
        try:
            g.delete()
        except Exception:
            pass
        g = db.select_graph(GRAPH_NAME)

    load_nodes(g, skb, emb, label_map)
    create_indexes(g, label_map)
    load_edges(g, skb, rel_map)

    fin_n, fin_e = graph_counts(g)
    print(f"[verify] final graph: {fin_n} nodes / {fin_e} edges (expected {exp_nodes}/{exp_edges})")
    assert fin_n == exp_nodes, f"node count mismatch {fin_n} != {exp_nodes}"
    assert fin_e == exp_edges, f"edge count mismatch {fin_e} != {exp_edges}"

    # Force a durable snapshot so the load survives container restart/removal.
    try:
        db.connection.bgsave()
        print("[persist] BGSAVE triggered")
    except Exception as ex:
        print(f"[persist] bgsave note: {ex}")

    print(f"\nETL OK in {time.time()-t_all:.1f}s total")


if __name__ == "__main__":
    main()
