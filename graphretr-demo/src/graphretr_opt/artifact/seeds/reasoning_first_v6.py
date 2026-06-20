"""Seed for the run-6 'reasoning_first_v6' family: the run-5 reasoning-rerank arm
plus the new recall operator G.reformulate. extract routes the query, typed
dense retrieval builds a pool, fixed-prompt G.llm_rerank reorders it, and -- only
when the first hop looks WEAK (gold likely not generated) -- the query is
reformulated from the top retrieved context, re-retrieved, and the fresh pool is
merged and re-ranked against the ORIGINAL question.

The hop/stop policy is ordinary editable Python gated by the metered llm_calls
cost axis: "stop when one hop suffices" is not hardcoded -- it emerges from
recall-pressure (reformulate to find missing gold) vs cost-pressure (a wasted
LLM call is penalized). The optimizer evolves this branch logic in search.py
only; extract/llm_rerank/reformulate prompts are fixed (editable is run-7).
"""

WEAK = 0.5   # first-hop top rerank score below which gold is likely not generated


def search(q, G):
    info = G.extract(q)
    hits = G.vector_search(q, k=50, label=info["answer_label"])
    scores = {nid: sim for nid, sim in hits}

    pool = [nid for nid, _ in hits[:30]]
    rer = G.llm_rerank(q, pool, top=30)
    for nid, s in rer:
        scores[nid] = scores.get(nid, 0.0) + s

    # Weak first hop => attack RECALL (generation), not ranking: reformulate the
    # query from the best retrieved context and re-retrieve a fresh pool.
    if not rer or rer[0][1] < WEAK:
        ctx = [nid for nid, _ in rer[:10]] or pool[:10]
        q2 = G.reformulate(ctx, top=10)
        if q2 and q2 != q:
            fresh = G.vector_search(q2, k=30, label=info["answer_label"])
            for nid, sim in fresh:
                scores[nid] = scores.get(nid, 0.0) + 0.5 * sim
            new_ids = [nid for nid, _ in fresh if nid not in set(p for p in pool)]
            if new_ids:
                # judge the FRESH candidates against the ORIGINAL question
                for nid, s in G.llm_rerank(q, new_ids[:20], top=20):
                    scores[nid] = scores.get(nid, 0.0) + s
    return scores
