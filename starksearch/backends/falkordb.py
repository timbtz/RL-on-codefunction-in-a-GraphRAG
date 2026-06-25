"""FalkorDB backend. The read-only wall is free here: every call goes through
GRAPH.RO_QUERY, which the server rejects writes on.

Engine quirks this codebase depends on (verified on the live server):
  * vector `score` = cosine DISTANCE (0..2, lower = closer), pre-sorted
    ascending -- the docs' ORDER BY score DESC example is wrong.
  * algo.SPpaths with maxLen >= 4 runs away for minutes and IGNORES the query
    timeout -- callers must cap maxLen at 3 (primitives.shortest_path does).
"""
from falkordb import FalkorDB

from .base import GraphBackend


class FalkorDBBackend(GraphBackend):
    def __init__(self, host: str, port: int, graph_name: str):
        self._db = FalkorDB(host=host, port=port)
        self._g = self._db.select_graph(graph_name)
        self.query_count = 0

    def ro_query(self, query, params=None, timeout_ms=2000):
        self.query_count += 1
        return self._g.ro_query(query, params=params or {}, timeout=timeout_ms).result_set

    def configure(self, key, value):
        try:
            self._db.connection.execute_command("GRAPH.CONFIG", "SET", key, str(value))
            return True
        except Exception as ex:
            print(f"[backend] GRAPH.CONFIG SET {key} note: {ex}")
            return False
