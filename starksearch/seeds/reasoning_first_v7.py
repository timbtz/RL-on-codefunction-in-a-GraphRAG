"""Seed for the 'reasoning_first_v7' family -- the END-TO-END-EDITABLE arm.

Unlike v6 (which could only edit the control flow of search() and called the
retrieval/LLM operators as frozen G.* methods), v7 OWNS the whole pipeline. Every
operator below -- extract, vector_search, text_search, llm_rerank, reformulate,
graph_recall -- is an ordinary helper function defined HERE, in the editable
artifact. They are built on three raw, trusted primitives the sandbox exposes:

  * G.query(cypher, params)  -- read-only Cypher (writes rejected by the engine)
  * G.embed(text)            -- query embedding (same model the nodes are indexed)
  * G.llm(system, user, ...) -- one metered+cached JSON-mode LLM call

So the optimizer can rewrite the prompts, the Cypher, the fusion math, add new
operators, or replace any of these wholesale. The recall ceiling is now in scope:
if vector search cannot surface the gold, write a graph walk that does (see
graph_recall) -- the gold is often a few hops from the query's entities.

This seed STARTS from run-7's evolved strategy (recall@20 ~0.43 on test), ported
into the editable v7 form: the typed anchor-hop (+0.6) and the per-keyword
conjunction bridge (+0.9) -- run-7's two winning recall levers -- are now plain
editable helpers (get_neighbors / filter_nodes) built on raw G.query Cypher,
not frozen G.* methods. The optimizer inherits run-7's quality AND can now edit
every operator end to end.

Sandbox rules to keep edits valid: NO imports, NO underscore attributes, and only
these builtins -- len sorted set dict list tuple min max sum enumerate zip range
abs round float bool int str any all reversed. Catch errors with a BARE `except:`
(exception-type names are not available). G.llm always returns a dict.
"""

WEAK = 0.5   # first-hop top rerank score below which gold is likely not generated

EXTRACT_PROMPT = (
    "You are a query analyzer for a biomedical knowledge graph. Given one "
    "natural-language question, return ONLY a JSON object with these keys:\n"
    '  "keywords": up to 6 salient entity names/phrases copied from the question.\n'
    '  "anchor_ntypes": node types of the entities the question MENTIONS.\n'
    '  "answer_ntype": the single node type the ANSWER should be, or null.\n'
    '  "rel_types": up to 5 relation types likely connecting anchors to answers.\n'
    "Use ONLY these exact strings.\n"
    "node types: {ntypes}\nrelation types: {rels}"
)

RERANK_PROMPT = (
    "You are a relevance judge for a biomedical knowledge-graph search. Given one "
    "question and a numbered list of candidate nodes (name + type + details), "
    "score how likely EACH candidate is the entity the question asks for. Judge by "
    "RELATIONAL relevance and reasoning -- a correct answer can share few words "
    "with the question, and a wordy near-duplicate can be wrong. Return ONLY a JSON "
    'object {"scores": {"<index>": <float in 0..1>, ...}} with one entry per index.'
)

REFORMULATE_PROMPT = (
    "You rewrite a biomedical knowledge-graph search query when the first "
    "retrieval looked weak. Given the original question and the documents of its "
    "top retrieved nodes, write ONE improved search query that would surface the "
    "entity the question actually asks for: name the likely answer entity/type and "
    "the salient relationship in plain terms, drop question phrasing, add precise "
    "synonyms or the connecting concept the first pass missed. Do NOT invent facts. "
    'Return ONLY a JSON object {"query": "<the rewritten query, one line>"}.'
)


def _strlist(v, cap):
    """Tolerant: coerce v (expected a list of strings) to <=cap clean strings."""
    out = []
    try:
        for s in v:
            s = str(s).strip()[:80]
            if s:
                out.append(s)
            if len(out) >= cap:
                break
    except:
        pass
    return out


def extract(G, q):
    """LLM query analysis -> {keywords, anchor_ntypes, answer_ntype, answer_label,
    rel_types}, validated against the graph's allowlists. EDIT the prompt freely."""
    ntypes, labels, rels = set(G.ntypes), set(G.labels), set(G.rel_types)
    sys = EXTRACT_PROMPT.format(ntypes=sorted(ntypes), rels=sorted(rels))
    raw = G.llm(sys, q)
    ans = raw.get("answer_ntype")
    try:
        ans = ans if ans in ntypes else None
    except:
        ans = None
    label = ans.replace("/", "_") if ans else None
    return {
        "keywords": _strlist(raw.get("keywords"), 6),
        "anchor_ntypes": [t for t in _strlist(raw.get("anchor_ntypes"), 4)
                          if t in ntypes],
        "answer_ntype": ans,
        "answer_label": label if label in labels else None,
        "rel_types": [r for r in _strlist(raw.get("rel_types"), 5) if r in rels],
    }


def vector_search(G, text, k=50, label=None):
    """ANN by query-text embedding via the vector index. -> [(node_id, sim)] desc.
    label=None fans out over every per-type index and merges. EDIT ME."""
    vec = G.embed(text)
    labels = [label] if label else sorted(set(G.labels))
    merged = []
    for lb in labels:
        try:
            rows = G.query(
                "CALL db.idx.vector.queryNodes($label,'embedding',$k,vecf32($vec)) "
                "YIELD node, score RETURN node.id, score",
                {"label": lb, "k": k, "vec": vec})
        except:
            rows = []
        for r in rows:
            merged.append((int(r[0]), 1.0 - float(r[1])))   # distance -> similarity
    merged.sort(key=lambda t: t[1], reverse=True)
    return merged[:k]


def text_search(G, text, k=20):
    """Full-text (BM25) keyword search over node docs. -> [(node_id, score)] desc."""
    toks, seen = [], set()
    for w in "".join(c if c.isalnum() else " " for c in text.lower()).split():
        if len(w) >= 3 and w not in seen:
            seen.add(w)
            toks.append(w)
    toks = toks[:24]
    if not toks:
        return []
    expr = "|".join(toks)
    try:
        rows = G.query(
            "CALL db.idx.fulltext.queryNodes('Entity', $q) YIELD node, score "
            "RETURN node.id, score ORDER BY score DESC LIMIT " + str(int(k)),
            {"q": expr})
    except:
        return []
    return [(int(r[0]), float(r[1])) for r in rows]


def llm_rerank(G, q, ids, top=20):
    """Reasoning-relevance rerank of ids against q. -> [(node_id, 0..1)] desc.
    EDIT the prompt/scoring -- this is the main ranking lever."""
    pool, seen = [], set()
    for nid in ids:
        nid = int(nid)
        if nid not in seen and len(pool) < 50:
            seen.add(nid)
            pool.append(nid)
    if not pool:
        return []
    texts = G.get_text(pool, limit=len(pool))
    lines = "\n".join("[" + str(i) + "] " + str(texts.get(nid, ""))[:500]
                      for i, nid in enumerate(pool))
    raw = G.llm(RERANK_PROMPT, "Question: " + q + "\n\nCandidates:\n" + lines,
                max_tokens=1500)
    sc = raw.get("scores", {})
    out = []
    for i, nid in enumerate(pool):
        v = sc.get(str(i))
        try:
            out.append((nid, float(v)))
        except:
            pass
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:top]


def reformulate(G, q, ctx_ids, top=10):
    """Rewrite a weak query from its top retrieved docs. -> new query str or None."""
    pool = [int(x) for x in ctx_ids][:top]
    if not pool:
        return None
    texts = G.get_text(pool, limit=len(pool))
    lines = "\n".join(str(texts.get(nid, ""))[:500] for nid in pool)
    raw = G.llm(REFORMULATE_PROMPT, "Question: " + q + "\n\nTop documents:\n" + lines)
    q2 = raw.get("query")
    try:
        q2 = str(q2).strip()
    except:
        return None
    return q2 or None


def graph_recall(G, anchor_ids, answer_ntype=None, max_hops=2, max_nodes=200):
    """GRAPH-STRUCTURAL RECALL (the recall-ceiling lever). BFS out from the
    query's anchor nodes and keep answer-typed nodes -- surfaces gold that is a
    few hops away in the graph but NOT vector-similar. EDIT/EXTEND ME: try
    shortest_path between two anchors, typed edges, deeper hops, etc."""
    ids = [int(x) for x in anchor_ids][:20]
    if not ids:
        return []
    k = min(max(int(max_hops), 1), 3)
    try:
        rows = G.query(
            "UNWIND $ids AS i MATCH (s:Entity {id:i}) "
            "CALL algo.bfs(s, " + str(k) + ", null) YIELD nodes "
            "UNWIND nodes AS n RETURN DISTINCT n.id, n.ntype "
            "LIMIT " + str(int(max_nodes)),
            {"ids": ids})
    except:
        return []
    out = []
    for r in rows:
        if answer_ntype is None or str(r[1]) == answer_ntype:
            out.append(int(r[0]))
    return out


def get_neighbors(G, ids, rel_type=None, direction="both", limit=50):
    """1-hop edges of ids via raw Cypher. -> [(src_id, rel_type, dst_id)].
    direction in {out,in,both}; rel_type=None spans every edge type. This is the
    typed-reachability lever the anchor-hop and conjunction bridge ride on --
    EDIT the direction, rel budget, or limit to change what graph edges count."""
    nodes = [int(x) for x in ids][:200]
    if not nodes:
        return []
    rp = ("[r:`" + str(rel_type) + "`]") if rel_type else "[r]"
    if direction == "out":
        pat = "-" + rp + "->"
    elif direction == "in":
        pat = "<-" + rp + "-"
    else:
        pat = "-" + rp + "-"
    try:
        rows = G.query(
            "UNWIND $ids AS i MATCH (a:Entity {id:i})" + pat + "(b) "
            "RETURN a.id, type(r), b.id LIMIT " + str(int(limit)),
            {"ids": nodes})
    except:
        return []
    out = []
    for r in rows:
        try:
            out.append((int(r[0]), str(r[1]), int(r[2])))
        except:
            pass
    return out


def filter_nodes(G, ids, ntype=None, limit=200):
    """Keep only ids whose n.ntype == ntype (None => passthrough). -> [node_id].
    The answer-type gate that turns a broad neighbor set into typed candidates."""
    nodes = [int(x) for x in ids][:1000]
    if not nodes:
        return []
    if not ntype:
        return nodes[:limit]
    try:
        rows = G.query(
            "UNWIND $ids AS i MATCH (n:Entity {id:i}) WHERE n.ntype = $nt "
            "RETURN n.id LIMIT " + str(int(limit)),
            {"ids": nodes, "nt": ntype})
    except:
        return []
    return [int(r[0]) for r in rows]


def search(q, G):
    info = extract(G, q)
    label = info["answer_label"]

    hits = vector_search(G, q, k=50, label=label)
    scores = {}
    for nid, sim in hits:
        scores[nid] = 0.3 * sim

    pool = [nid for nid, _ in hits[:30]]
    rer = llm_rerank(G, q, pool, top=30)
    for nid, s in rer:
        scores[nid] = scores.get(nid, 0.0) + s

    anc_types = info["anchor_ntypes"]
    rel_types = info["rel_types"]
    ans_ntype = info["answer_ntype"]

    # (1) Typed anchor-hop. Route the extracted rel_types into 1-hop neighbors of
    #     the query's top anchors and keep answer-typed nodes. The +0.6 bonus
    #     dominates llm_rerank's [0,1] score, so gold reached via the RIGHT edge
    #     type outranks text-similar decoys. Broad BFS is a cross-type fallback
    #     only when typed expansion returns nothing. EDIT the bonus / hop policy.
    if anc_types and ans_ntype:
        anc_label = anc_types[0].replace("/", "_")
        anc_hits = vector_search(G, q, k=8, label=anc_label)
        anc_ids = [nid for nid, _ in anc_hits[:5]]
        if anc_ids:
            direct = set()
            for rt in rel_types[:4]:
                for _, _, dst in get_neighbors(G, anc_ids, rel_type=rt,
                                               direction="both", limit=50):
                    direct.add(dst)
            typed = filter_nodes(G, list(direct), ntype=ans_ntype, limit=100) if direct else []
            for nid in typed:
                scores[nid] = scores.get(nid, 0.0) + 0.6
            if not typed and anc_types[0] != ans_ntype:
                for nid in graph_recall(G, anc_ids, answer_ntype=ans_ntype,
                                        max_hops=2, max_nodes=120):
                    scores[nid] = scores.get(nid, 0.0) + 0.25

    # (2) Per-keyword conjunction bridge. Resolve EACH keyword to its own anchor
    #     (label=None spans node types), expand via rel_types, filter to
    #     answer_ntype, and award the INTERSECTION a +0.9 bonus -- candidates
    #     satisfying EVERY constraint outrank single-side dominators, and
    #     stacked-constraint gold the dense top-50 missed gets seeded in. EDIT ME.
    kws = info["keywords"]
    if len(kws) >= 2 and ans_ntype:
        per_kw_sets = []
        for kw in kws[:3]:
            kw_hits = vector_search(G, kw, k=5)
            kw_ids = [nid for nid, _ in kw_hits[:3]]
            if not kw_ids:
                continue
            nb = set()
            for rt in rel_types[:4]:
                for _, _, dst in get_neighbors(G, kw_ids, rel_type=rt,
                                               direction="both", limit=50):
                    nb.add(dst)
            if not nb:
                continue
            tk = set(filter_nodes(G, list(nb), ntype=ans_ntype, limit=100))
            if tk:
                per_kw_sets.append(tk)
        for kset in per_kw_sets:
            for nid in kset:
                scores[nid] = scores.get(nid, 0.0) + 0.3
        if len(per_kw_sets) >= 2:
            bridge = per_kw_sets[0]
            for kset in per_kw_sets[1:]:
                bridge = bridge & kset
            for nid in bridge:
                scores[nid] = scores.get(nid, 0.0) + 0.9

    # (3) Weak first hop => attack RECALL (generation): reformulate from the best
    #     context, re-retrieve, judge the fresh pool against the ORIGINAL question.
    #     Strong hop => one relative rerank pass over the merged top-20 to resolve
    #     intersection-vs-container ties additive scoring cannot.
    if not rer or rer[0][1] < WEAK:
        ctx = [nid for nid, _ in rer[:10]] or pool[:10]
        q2 = reformulate(G, q, ctx, top=10)
        if q2 and q2 != q:
            fresh = vector_search(G, q2, k=30, label=label)
            for nid, sim in fresh:
                scores[nid] = scores.get(nid, 0.0) + 0.5 * sim
            seen = set(pool)
            new_ids = [nid for nid, _ in fresh if nid not in seen]
            if new_ids:
                for nid, s in llm_rerank(G, q, new_ids[:20], top=20):
                    scores[nid] = scores.get(nid, 0.0) + s
    else:
        top_final = [nid for nid, _ in sorted(scores.items(), key=lambda x: -x[1])[:20]]
        if len(top_final) > 1:
            for nid, s in llm_rerank(G, q, top_final, top=20):
                scores[nid] = scores.get(nid, 0.0) + s

    return scores
