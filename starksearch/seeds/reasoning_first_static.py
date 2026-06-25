"""Control arm for 'reasoning_first': structurally identical, but the rerank
step is the no-LLM cosine G.rank_by_text instead of G.llm_rerank. The
ablation delta reasoning_first - reasoning_first_static isolates the value of
the reasoning signal from the candidate-generation + graph-expansion structure
the two share.
"""

WEAK = 0.5  # first-hop top cosine score below which we take a second hop


def search(q, G):
    info = G.extract(q)
    hits = G.vector_search(q, k=50, label=info["answer_label"])
    scores = {nid: sim for nid, sim in hits}

    pool = [nid for nid, _ in hits[:30]]
    rer = G.rank_by_text(pool, q, top=30)
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
            for nid, s in G.rank_by_text(fresh[:30], q, top=20):
                scores[nid] = s
    return scores
