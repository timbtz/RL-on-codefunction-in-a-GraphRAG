"""Seed for the 'extract_first' strategy family: G.extract routes the query
(answer type + likely relations), then typed dense retrieval plus a
relation-guided 1-hop expansion re-ranked against the query. The principled
replacement for run1/run2's hardcoded keyword tables -- the query
understanding lives in the LLM call, not in this source.
"""


def search(q, G):
    info = G.extract(q)
    hits = G.vector_search(q, k=60, label=info["answer_label"])
    scores = {nid: sim for nid, sim in hits}

    anchors = [nid for nid, _ in hits[:10]]
    fresh = []
    for rel in info["rel_types"]:
        for a, r, b in G.get_neighbors(anchors, rel_type=rel,
                                       direction="both", limit=80):
            if b not in scores:
                fresh.append(b)
    if fresh:
        ranked = G.rank_by_text(fresh[:200], q, top=40)
        keep = None
        if info["answer_ntype"] and ranked:
            keep = set(G.filter_nodes([nid for nid, _ in ranked],
                                      ntype=info["answer_ntype"]))
        for nid, sim in ranked:
            if keep is None or nid in keep:
                scores[nid] = sim * 0.9
    return scores
