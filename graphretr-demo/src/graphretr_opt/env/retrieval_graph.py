"""RetrievalGraph -- the object the candidate program sees as `G`.

Holds one read-only backend connection, a primitive cache, and the query
embedder, and exposes the seven-method closed DSL (layer 1, immutable). Every
method: validates args via primitives.py, emits ONE parameterized capped query
through the backend's ro_query, returns node-id-keyed plain-Python data (never
raw graph objects), and memoizes the result.

All collaborators are underscore-prefixed: the sandbox AST-gate forbids any
`._x` attribute access in candidate code, so the program's only surface is the
public methods + the read-only allowlist properties + describe().
"""
import numpy as np

from .cache import PrimitiveCache
from .embedder import QueryEmbedder
from . import primitives as P


class RetrievalGraph:
    def __init__(self, cfg, backend, cache: PrimitiveCache = None,
                 embedder: QueryEmbedder = None):
        self._cfg = cfg
        self._backend = backend
        self._cache = cache or PrimitiveCache()
        self._embedder = embedder or QueryEmbedder()

        # One-time engine safety config (global to our container alone).
        backend.configure("TIMEOUT_MAX", 10_000)
        backend.configure("QUERY_MEM_CAPACITY", 268_435_456)

        labels = sorted(r[0] for r in self._ro(
            "CALL db.labels() YIELD label RETURN label"))
        rels = sorted(r[0] for r in self._ro(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"))
        ntypes = sorted(r[0] for r in self._ro(
            "MATCH (n:Entity) RETURN DISTINCT n.ntype"))
        self._allow = P.Allowlists(
            labels=[l for l in labels if l != "Entity"],  # vector-indexed per-type labels
            rel_types=rels, ntypes=ntypes)

    # ---------------------------------------------------------------- plumbing

    def _ro(self, q, params=None):
        return self._backend.ro_query(q, params=params, timeout_ms=self._cfg.query_timeout_ms)

    def _cap(self, x, name, hi):
        return P.clamp(x, name, hi)

    # -------------------------------------------------------------- primitives

    def vector_search(self, text, k=20, label=None):
        """Top-k ANN by query-text embedding. -> [(node_id, similarity)], cosine
        similarity in ~[0,1], sorted descending. label=None fans out over all 10
        per-type labels and merges client-side."""
        P.nonempty_str(text, "text")
        k = self._cap(k, "k", self._cfg.max_fanout)
        labels = ((self._allow.check_label(label),) if label is not None
                  else self._allow.labels)
        key = ("vector_search", text, k, labels)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        vec = self._embedder.encode(text).tolist()
        merged = []
        for lb in labels:
            rows = self._ro(
                f"CALL db.idx.vector.queryNodes($label,'embedding',{k},vecf32($vec)) "
                "YIELD node, score RETURN node.id, score",
                {"label": lb, "vec": vec})
            # score is cosine DISTANCE (lower = closer); convert to similarity.
            merged.extend((int(r[0]), 1.0 - float(r[1])) for r in rows)
        merged.sort(key=lambda t: t[1], reverse=True)
        return list(self._cache.put(key, merged[:k]))

    def get_neighbors(self, ids, rel_type=None, direction="out", limit=50):
        """1-hop edges of ids. -> [(src_id, rel_type, dst_id)], at most `limit`."""
        ids = P.ids_in(ids, self._cfg.max_fanout)
        rel = self._allow.check_rel(rel_type)
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in' or 'both'")
        limit = self._cap(limit, "limit", self._cfg.max_fanout)
        rp = f"[r:`{rel}`]" if rel else "[r]"
        pat = {"out": f"-{rp}->", "in": f"<-{rp}-", "both": f"-{rp}-"}[direction]
        key = ("get_neighbors", tuple(ids), rel, direction, limit)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        rows = self._ro(
            f"UNWIND $ids AS i MATCH (a:Entity {{id:i}}){pat}(b) "
            f"RETURN a.id, type(r), b.id LIMIT {limit}",
            {"ids": ids})
        out = [(int(r[0]), str(r[1]), int(r[2])) for r in rows]
        return list(self._cache.put(key, out))

    def k_hop_expand(self, ids, k=2, rel_type=None, max_nodes=200):
        """BFS out to depth k (algo.bfs: dedups during traversal, no path blowup).
        -> [node_id] (may include the start ids), at most `max_nodes`."""
        ids = P.ids_in(ids, 50)
        k = self._cap(k, "k", self._cfg.max_k)
        rel = self._allow.check_rel(rel_type)
        max_nodes = self._cap(max_nodes, "max_nodes", self._cfg.max_fanout)
        key = ("k_hop_expand", tuple(ids), k, rel, max_nodes)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        rt = "$rt" if rel else "null"
        rows = self._ro(
            f"UNWIND $ids AS i MATCH (s:Entity {{id:i}}) "
            f"CALL algo.bfs(s, {k}, {rt}) YIELD nodes "
            f"UNWIND nodes AS n RETURN DISTINCT n.id LIMIT {max_nodes}",
            {"ids": ids, **({"rt": rel} if rel else {})})
        out = [int(r[0]) for r in rows]
        return list(self._cache.put(key, out))

    def filter_nodes(self, ids, ntype=None, text_contains=None, limit=200):
        """Keep only ids matching node-type and/or case-insensitive text substring.
        -> [node_id], at most `limit`."""
        ids = P.ids_in(ids)
        limit = self._cap(limit, "limit", self._cfg.max_fanout)
        conds, params = [], {"ids": ids}
        if ntype is not None:
            conds.append("n.ntype = $nt")
            params["nt"] = self._allow.check_ntype(ntype)
        if text_contains is not None:
            P.nonempty_str(text_contains, "text_contains")
            conds.append("toLower(n.text) CONTAINS toLower($tc)")
            params["tc"] = text_contains
        where = ("WHERE " + " AND ".join(conds) + " ") if conds else ""
        key = ("filter_nodes", tuple(ids), ntype, text_contains, limit)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        rows = self._ro(
            f"UNWIND $ids AS i MATCH (n:Entity {{id:i}}) {where}"
            f"RETURN n.id LIMIT {limit}",
            params)
        out = [int(r[0]) for r in rows]
        return list(self._cache.put(key, out))

    def shortest_path(self, a, b, max_len=3):
        """Shortest undirected path a -> b (algo.SPpaths, length-optimized).
        -> [node_id] including both endpoints, or [] if none within max_len.
        max_len is hard-capped at 3: SPpaths at maxLen>=4 runs away for tens of
        seconds on this graph and IGNORES the query timeout, so the cap is the
        wall."""
        a = P.as_int(a, "a")
        b = P.as_int(b, "b")
        max_len = self._cap(max_len, "max_len", 3)
        key = ("shortest_path", a, b, max_len)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        rows = self._ro(
            "MATCH (a:Entity {id:$a}), (b:Entity {id:$b}) "
            f"CALL algo.SPpaths({{sourceNode:a, targetNode:b, relDirection:'both', "
            f"maxLen:{max_len}, pathCount:1}}) "
            "YIELD path RETURN [n IN nodes(path) | n.id]",
            {"a": a, "b": b})
        out = [int(x) for x in rows[0][0]] if rows else []
        return list(self._cache.put(key, out))

    def rank_by_text(self, ids, query_text, top=20):
        """Re-rank the given ids by embedding similarity to query_text (query is
        embedded once, node embeddings fetched by id). -> [(node_id, similarity)]
        sorted descending, at most `top`."""
        ids = P.ids_in(ids)
        P.nonempty_str(query_text, "query_text")
        top = self._cap(top, "top", self._cfg.max_fanout)
        key = ("rank_by_text", tuple(ids), query_text, top)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        qv = self._embedder.encode(query_text)
        rows = self._ro(
            "UNWIND $ids AS i MATCH (n:Entity {id:i}) RETURN n.id, n.embedding",
            {"ids": ids})
        sims = [(int(r[0]), float(np.dot(qv, np.asarray(r[1], dtype=np.float32))))
                for r in rows]
        sims.sort(key=lambda t: t[1], reverse=True)
        return list(self._cache.put(key, sims[:top]))

    def get_text(self, ids, limit=50):
        """Fetch node documents. -> {node_id: text}, at most `limit` entries."""
        ids = P.ids_in(ids)
        limit = self._cap(limit, "limit", self._cfg.max_fanout)
        ids = ids[:limit]
        key = ("get_text", tuple(ids))
        hit = self._cache.get(key)
        if hit is not None:
            return dict(hit)
        rows = self._ro(
            "UNWIND $ids AS i MATCH (n:Entity {id:i}) RETURN n.id, n.text",
            {"ids": ids})
        out = {int(r[0]): str(r[1]) for r in rows}
        return dict(self._cache.put(key, out))

    # ------------------------------------------------------------- introspection

    @property
    def labels(self):
        """Vector-indexed per-type labels (valid `label=` args)."""
        return self._allow.labels

    @property
    def rel_types(self):
        """Valid `rel_type=` args."""
        return self._allow.rel_types

    @property
    def ntypes(self):
        """Valid `ntype=` args (exact STaRK node-type strings)."""
        return self._allow.ntypes

    def describe(self):
        """Primitive API doc with the live allowlists -- used in the reflection
        prompt and available to candidate programs."""
        cfg = self._cfg
        return f"""\
G is a RetrievalGraph over the STaRK-prime biomedical knowledge graph
(129,375 nodes / 8,100,498 directed typed edges). All methods are read-only,
capped (fan-out <= {cfg.max_fanout}, k-hop depth <= {cfg.max_k}), and each underlying graph query
has a {cfg.query_timeout_ms} ms timeout (a timeout raises -- catch nothing, just keep fan-out
modest). Invalid arguments raise ValueError.

G.vector_search(text, k=20, label=None) -> list[(node_id:int, similarity:float)]
    ANN over node-text embeddings (all-MiniLM-L6-v2, cosine). similarity ~[0,1],
    sorted descending. label=None searches all 10 node-type labels and merges;
    label='disease' (etc.) restricts to one type.
G.get_neighbors(ids, rel_type=None, direction='out', limit=50) -> list[(src_id, rel_type:str, dst_id)]
    1-hop edges of the given node id(s). direction: 'out'|'in'|'both'.
G.k_hop_expand(ids, k=2, rel_type=None, max_nodes=200) -> list[node_id]
    BFS neighborhood out to depth k (deduplicated; includes start ids).
G.filter_nodes(ids, ntype=None, text_contains=None, limit=200) -> list[node_id]
    Keep ids whose node matches the exact ntype and/or whose document contains
    the substring (case-insensitive).
G.shortest_path(a, b, max_len=3) -> list[node_id]
    Shortest undirected path between two ids ([] if none within max_len,
    which is hard-capped at 3).
G.rank_by_text(ids, query_text, top=20) -> list[(node_id, similarity)]
    Re-rank candidate ids by embedding similarity to query_text, descending.
G.get_text(ids, limit=50) -> dict[node_id, str]
    Node documents (name + type + details), for inspection/filtering.

Allowlists (use these EXACT strings):
  label / node-type labels: {list(self._allow.labels)}
  ntype strings:            {list(self._allow.ntypes)}
  rel_type strings:         {list(self._allow.rel_types)}
"""
