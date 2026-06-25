"""GraphBackend -- the one interface a graph engine must implement.

Swapping engines (FalkorDB -> Neo4j -> ...) is one new class with these two
methods; primitives.py and everything above it stay untouched.
"""
from abc import ABC, abstractmethod


class GraphBackend(ABC):
    #: monotonically increasing count of queries issued (read by RunStats).
    query_count: int = 0

    @abstractmethod
    def ro_query(self, query: str, params: dict | None = None,
                 timeout_ms: int = 2000) -> list[list]:
        """Run a READ-ONLY query; the engine must reject writes server-side.
        Returns rows as lists (engine-native row shape)."""

    def configure(self, key: str, value) -> bool:
        """Best-effort engine-level safety knob (timeouts, memory caps)."""
        return False
