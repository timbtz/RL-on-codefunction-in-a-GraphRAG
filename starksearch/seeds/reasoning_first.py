"""Seed for the 'reasoning_first' strategy family: G.extract routes the query,
typed dense retrieval builds a candidate pool, and the fixed-prompt
G.llm_rerank reasons about relational relevance to reorder it -- the lever that
can promote a text-dissimilar correct answer above a wordy decoy. The second
hop (graph-expand the rerank winners, then rerank the expansion) is taken only
when the first hop looks weak, so a wasted LLM call is penalized by the metered
llm_calls cost axis. The optimizer evolves this branch logic in search.py only.
"""

WEAK = 0.5  # first-hop top rerank score below which we take a second hop


def search(q, G):
    info = G.extract(q)
    hits = G.vector_search(q, k=50, label=info["answer_label"])
    scores = {nid: sim for nid, sim in hits}

    pool = [nid for nid, _ in hits[:30]]
    rer = G.llm_rerank(q, pool, top=30)
    for nid, s in rer:
        scores[nid] = scores.get(nid, 0.0) + s

    if not rer or rer[0][1] < WEAK:
        top_ids = [nid for nid, _ in rer[:5]] or pool[:5]
        fresh = []
        for rel in info["rel_types"]:
            for a, r, b in G.get_neighbors(top_ids, rel_type=rel,
                                           direction="both", limit=60):
                if b not in scores:
                    fresh.append(b)
        if fresh:
            for nid, s in G.llm_rerank(q, fresh[:30], top=20):
                scores[nid] = s
    return scores
