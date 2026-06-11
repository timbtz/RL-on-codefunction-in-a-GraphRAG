"""Vector seeds + anchor-neighborhood expansion with common-neighbor (intersection) boosting."""


def search(q, G):
    seeds = G.vector_search(q, k=100)
    scores = {nid: float(sim) for nid, sim in seeds}
    seen = set(scores)

    # Top seeds act as anchors; expand their 1-hop neighborhood both directions.
    anchors = [nid for nid, sim in seeds[:12]]
    freq = {}
    for src, rel, dst in G.get_neighbors(anchors, direction='both', limit=50):
        for x in (src, dst):
            freq[x] = freq.get(x, 0) + 1

    # Graph-only candidates: score by text relevance plus an intersection bonus
    # (freq>=2 == reached from multiple anchors -> "common target / parent-child" answers).
    new = [x for x in freq if x not in seen]
    text = {}
    if new:
        for nid, sim in G.rank_by_text(new, q, top=60):
            text[nid] = float(sim)
    for x in new:
        scores[x] = text.get(x, 0.0) * 0.5 + 0.1 * min(max(freq[x] - 1, 0), 3)

    # Seeds that are also graph hubs for the query's anchors get a small lift.
    for x in seen:
        if x in freq:
            scores[x] = scores[x] + 0.04 * min(freq[x], 4)

    return scores
