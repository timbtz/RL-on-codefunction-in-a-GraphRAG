"""Neo4j backend -- STAGE-2 SEAM (not used by the Stage-1 demo).

Porting notes for whoever fills this in:
  * ro_query -> session.execute_read with a literal-free parameterized query;
    return [list(record.values()) for record in result].
  * Vector search -> db.index.vector.queryNodes (Neo4j 5.x), which returns
    SIMILARITY (higher = closer) -- the FalkorDB code path converts distance
    to similarity, so primitives.vector_search's `1 - score` flip must become
    backend-specific (move the flip into the backend or a per-backend flag).
  * algo.bfs / algo.SPpaths -> APOC or GDS equivalents.
"""
from .base import GraphBackend


class Neo4jBackend(GraphBackend):
    def __init__(self, *a, **kw):
        raise NotImplementedError("Stage-2 seam: see module docstring")

    def ro_query(self, query, params=None, timeout_ms=2000):
        raise NotImplementedError
