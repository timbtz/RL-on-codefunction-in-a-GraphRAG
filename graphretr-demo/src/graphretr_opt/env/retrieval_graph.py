"""RetrievalGraph -- the object the candidate program sees as `G`.

Holds one read-only backend connection, a primitive cache, and the query
embedder, and exposes the closed DSL (layer 1, immutable). Every method:
validates args via primitives.py, emits ONE parameterized capped query
through the backend's ro_query, returns node-id-keyed plain-Python data (never
raw graph objects), and memoizes the result.

G.extract is the one LLM-backed primitive: a FIXED-prompt, fixed-schema
query-understanding call. It only accepts the exact query currently being
executed (the sandbox pins it), so a program cannot loop it over node texts,
and its result is cached in memory AND on disk -- at most one billed call per
unique query, ever. The prompt is not program-editable: the query->types
mapping cannot be overfitted into candidate source the way run1/run2's
hardcoded keyword arrays were.

All collaborators are underscore-prefixed: the sandbox AST-gate forbids any
`._x` attribute access in candidate code, so the program's only surface is the
public methods + the read-only allowlist properties + describe().
"""
import json
import os

import numpy as np

from .cache import PrimitiveCache
from .embedder import QueryEmbedder
from . import primitives as P

_EXTRACT_SYSTEM = """\
You are a query analyzer for a biomedical knowledge graph. Given one natural-
language question, return ONLY a JSON object with these keys:
  "keywords":      up to 6 salient entity names/phrases copied from the question
                   (the things to look up; no question words).
  "anchor_ntypes": node types of the entities the question MENTIONS.
  "answer_ntype":  the single node type the ANSWER should be, or null.
  "rel_types":     up to 5 relation types likely connecting anchors to answers.
Use ONLY these exact strings.
node types: {ntypes}
relation types: {rel_types}"""

_RERANK_SYSTEM = """\
You are a relevance judge for a biomedical knowledge-graph search. Given one
question and a numbered list of candidate nodes (name + type + details), score
how likely EACH candidate is the entity the question asks for. Judge by
RELATIONAL relevance and reasoning -- a correct answer can share few words with
the question, and a wordy near-duplicate can be wrong. Return ONLY a JSON
object {"scores": {"<index>": <float in 0..1>, ...}} with one entry per index."""


class RetrievalGraph:
    def __init__(self, cfg, backend, cache: PrimitiveCache = None,
                 embedder: QueryEmbedder = None, llm_budget=None):
        self._cfg = cfg
        self._backend = backend
        self._cache = cache or PrimitiveCache()
        self._embedder = embedder or QueryEmbedder()
        self._llm_budget = llm_budget
        self._llm_calls = 0          # extract() invocations; sandbox reads the delta
        self._pinned_query = None    # set by Sandbox.run; extract() only accepts this
        self._extract_disk = os.path.join(cfg.runs_dir, "extract_cache.json")
        self._extract_mem = None     # lazy-loaded disk cache
        self._rerank_disk = os.path.join(cfg.runs_dir, "llm_rerank_cache.json")
        self._rerank_mem = None      # lazy-loaded disk cache

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

    def _pin_query(self, query):
        """Trusted-caller hook (Sandbox.run): the only text extract() accepts."""
        self._pinned_query = query

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

    def text_search(self, text, k=20):
        """Full-text (BM25-style) keyword search over node documents.
        -> [(node_id, score)], score descending, at most k. The text is
        tokenized to alphanumeric terms OR-ed together; punctuation and
        sub-3-char tokens are dropped."""
        P.nonempty_str(text, "text")
        k = self._cap(k, "k", self._cfg.max_fanout)
        toks, seen = [], set()
        for w in "".join(c if c.isalnum() else " " for c in text.lower()).split():
            if len(w) >= 3 and w not in seen:
                seen.add(w)
                toks.append(w)
        toks = toks[:24]
        if not toks:
            return []
        expr = "|".join(toks)
        key = ("text_search", expr, k)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        rows = self._ro(
            "CALL db.idx.fulltext.queryNodes('Entity', $q) YIELD node, score "
            f"RETURN node.id, score ORDER BY score DESC LIMIT {k}",
            {"q": expr})
        out = [(int(r[0]), float(r[1])) for r in rows]
        return list(self._cache.put(key, out))

    def extract(self, text):
        """LLM query analysis (fixed prompt, fixed schema). Accepts ONLY the
        current query q. -> {'keywords': [str], 'anchor_ntypes': [str],
        'answer_ntype': str|None, 'rel_types': [str]} -- type/relation fields
        validated against the allowlists (invalid values dropped, never raised).
        Cached on disk: repeat queries are free."""
        P.nonempty_str(text, "text")
        if text != self._pinned_query:
            raise ValueError("extract() accepts only the current query q")
        if self._llm_budget is None:
            raise ValueError("extract() is not enabled (no OPENAI_API_KEY)")
        self._llm_calls += 1
        if self._extract_mem is None:
            self._extract_mem = (json.load(open(self._extract_disk))
                                 if os.path.exists(self._extract_disk) else {})
        hit = self._extract_mem.get(text)
        if hit is not None:
            return json.loads(json.dumps(hit))  # defensive copy
        system = _EXTRACT_SYSTEM.format(ntypes=list(self._allow.ntypes),
                                        rel_types=list(self._allow.rel_types))
        raw = self._llm_budget.chat_json(system, text,
                                         model=self._cfg.extract_model)
        out = self._validate_extract(raw)
        self._extract_mem[text] = out
        os.makedirs(os.path.dirname(self._extract_disk), exist_ok=True)
        json.dump(self._extract_mem, open(self._extract_disk, "w"))
        return json.loads(json.dumps(out))

    def _validate_extract(self, raw):
        def strs(v, cap):
            v = v if isinstance(v, list) else []
            return [s.strip()[:80] for s in v
                    if isinstance(s, str) and s.strip()][:cap]
        ans = raw.get("answer_ntype") if isinstance(raw, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        ans = ans if ans in self._allow.ntypes else None
        # ntype 'gene/protein' <-> vector-index label 'gene_protein'
        label = ans.replace("/", "_") if ans else None
        return {
            "keywords": strs(raw.get("keywords"), 6),
            "anchor_ntypes": [t for t in strs(raw.get("anchor_ntypes"), 4)
                              if t in self._allow.ntypes],
            "answer_ntype": ans,
            "answer_label": label if label in self._allow.labels else None,
            "rel_types": [r for r in strs(raw.get("rel_types"), 5)
                          if r in self._allow.rel_types],
        }

    def llm_rerank(self, query, ids, top=20):
        """Reasoning-relevance rerank of candidate ids against the current query
        (fixed prompt, fixed schema). Accepts ONLY the current query q (the
        sandbox pins it). The pool is deduped, sorted and hard-capped at
        cfg.rerank_pool_max, then sent to the LLM with each node's document.
        -> [(node_id, score in 0..1)] sorted descending, at most `top`; ids the
        model did not score are omitted. Cached on disk by (query, pool): repeat
        (query, pool) pairs are free and resume-safe. Metered + costed exactly
        like extract()."""
        P.nonempty_str(query, "query")
        if query != self._pinned_query:
            raise ValueError("llm_rerank() accepts only the current query q")
        if self._llm_budget is None:
            raise ValueError("llm_rerank() is not enabled (no OPENAI_API_KEY)")
        # Dedup preserving caller order: the candidates reach the LLM in the
        # order search() ranked them (e.g. dense-rank), so the model keeps that
        # prior. The cache key is order-INDEPENDENT (sorted), so the same pool
        # in any order is one billed call.
        seen, pool = set(), []
        for nid in P.ids_in(ids):
            if nid not in seen and len(pool) < self._cfg.rerank_pool_max:
                seen.add(nid)
                pool.append(nid)
        top = self._cap(top, "top", self._cfg.max_fanout)
        self._llm_calls += 1
        if self._rerank_mem is None:
            self._rerank_mem = (json.load(open(self._rerank_disk))
                                if os.path.exists(self._rerank_disk) else {})
        ckey = query + "\x00" + ",".join(str(i) for i in sorted(pool))
        scores = self._rerank_mem.get(ckey)
        if scores is None:
            texts = self.get_text(pool, limit=len(pool))
            lines = "\n".join(f"[{i}] {texts.get(nid, '')[:500]}"
                              for i, nid in enumerate(pool))
            raw = self._llm_budget.chat_json(
                _RERANK_SYSTEM, f"Question: {query}\n\nCandidates:\n{lines}",
                model=self._cfg.rerank_model, max_tokens=1500)
            sc = raw.get("scores", {}) if isinstance(raw, dict) else {}
            scores = {}
            for i, nid in enumerate(pool):
                v = sc.get(str(i))
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    scores[str(nid)] = float(v)
            self._rerank_mem[ckey] = scores
            os.makedirs(os.path.dirname(self._rerank_disk), exist_ok=True)
            json.dump(self._rerank_mem, open(self._rerank_disk, "w"))
        ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)
        return [(int(i), s) for i, s in ranked[:top]]

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
    ANN over node-text embeddings (cosine). similarity ~[0,1], sorted
    descending. label=None searches all 10 node-type labels and merges;
    label='disease' (etc.) restricts to one type.
G.text_search(text, k=20) -> list[(node_id:int, score:float)]
    Full-text (BM25-style) keyword search over node documents, score
    descending. Complements vector_search: exact names/rare terms score high
    here even when embeddings miss them. Combine both with rank fusion.
G.extract(q) -> dict
    LLM analysis of the query (accepts ONLY the exact `q` your search() was
    called with; cached, so repeat calls are free). Returns
    {{'keywords': [str], 'anchor_ntypes': [str], 'answer_ntype': str|None,
    'answer_label': str|None, 'rel_types': [str]}} -- all type/relation values
    already validated against the allowlists below (answer_label is the
    vector_search `label=` form of answer_ntype). Use it to pick vector_search
    labels, filter_nodes ntypes and get_neighbors/k_hop_expand rel_types
    instead of hardcoding keyword tables.
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
G.llm_rerank(q, ids, top=20) -> list[(node_id, score)]
    LLM reasoning-relevance rerank of a BOUNDED candidate pool against the
    query (accepts ONLY the exact `q`; pool hard-capped at {cfg.rerank_pool_max},
    cached by (q, pool) so repeats are free). score in [0,1], descending; ids
    the model did not score are dropped. Unlike rank_by_text it judges by
    reasoning, so it can rank a text-dissimilar correct answer above a wordy
    decoy -- but each unique (q, pool) costs one billed LLM call, so rerank a
    small dense top-k, not the whole graph.
G.get_text(ids, limit=50) -> dict[node_id, str]
    Node documents (name + type + details), for inspection/filtering.

Allowlists (use these EXACT strings):
  label / node-type labels: {list(self._allow.labels)}
  ntype strings:            {list(self._allow.ntypes)}
  rel_type strings:         {list(self._allow.rel_types)}
"""
