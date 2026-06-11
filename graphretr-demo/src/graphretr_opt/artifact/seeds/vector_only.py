"""Seed program for the 'vector_only' strategy family: naive vector-only
retrieval (the frozen baseline). The optimizer mutates copies; this never
changes.
"""


def search(q, G):
    hits = G.vector_search(q, k=100)
    return {nid: sim for nid, sim in hits}
