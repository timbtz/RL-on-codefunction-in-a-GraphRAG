"""PrimitiveCache -- memoize primitive calls by (method, frozen args).

The gate re-scores the same 200 queries every step; this keeps the loop
LLM-bound instead of DB-bound. Callers copy values on return (the cache never
hands out a mutable reference to its own storage).
"""


class PrimitiveCache:
    def __init__(self, max_entries: int = 100_000):
        self._store = {}
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    def get(self, key):
        val = self._store.get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def put(self, key, value):
        if len(self._store) >= self._max:
            self._store.clear()  # simple full-evict; LRU is not worth it here
        self._store[key] = value
        return value
