"""Seed for the 'hybrid_rrf' strategy family: dense (vector) + sparse
(full-text) retrieval fused by reciprocal-rank fusion -- the classic hybrid
baseline. No LLM, no graph traversal; what the optimizer adds on top is the
experiment.
"""

RRF_K = 60


def search(q, G):
    dense = G.vector_search(q, k=100)
    sparse = G.text_search(q, k=100)
    scores = {}
    for ranking in (dense, sparse):
        for rank, (nid, _) in enumerate(ranking):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores
