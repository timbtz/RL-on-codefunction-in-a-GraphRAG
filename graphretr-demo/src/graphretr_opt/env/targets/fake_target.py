"""FakeSearchTarget -- the in-memory SearchTarget adapter.

Deterministic canned context per query and a fixed CostMeter, so Phases 1-3 (the
whole optimizer loop, the MCQ reward, the recap table) run with zero infra -- no
Neo4j, no API keys, no subprocess. Two adapters behind one port = a real seam;
swapping in SubprocessSearchTarget is a one-line change at the wiring site.

The fake never inspects the FileSet beyond its sha (so two different candidates
return distinguishable-but-deterministic context), which is exactly what a loop
smoke test needs: the score is a pure function of (query, candidate), no I/O.
"""
from ..search_target import CostMeter, SearchResult


class FakeSearchTarget:
    def __init__(self, contexts=None, cost=None, fixed_doc="doc:FAKE",
                 vary_by_candidate=True):
        """contexts: optional {query -> context} to drive specific test
            scenarios (retrieval hit/miss, answerable/unanswerable). When a query
            is absent the fake echoes the query plus `fixed_doc`.
        cost: a CostMeter returned for every query (default: a small fixed cost).
        vary_by_candidate: append the FileSet sha so different candidates yield
            different (still deterministic) context -- lets a loop smoke test see
            score movement across mutations.
        """
        self._contexts = dict(contexts or {})
        self._cost = cost or CostMeter(llm_calls=2, tokens_in=100, tokens_out=20,
                                       db_queries=3, latency_s=0.01)
        self._fixed_doc = fixed_doc
        self._vary = vary_by_candidate

    def run(self, file_set, queries, timeout_s: float) -> dict:
        sha = getattr(file_set, "sha", "nosha")[:8]
        out = {}
        for q in queries:
            if q in self._contexts:
                ctx = self._contexts[q]
            else:
                tag = f" [{sha}]" if self._vary else ""
                ctx = f"[{self._fixed_doc}]\nFAKE CONTEXT for: {q}{tag}\n"
            out[q] = SearchResult(context=ctx, cost=self._cost)
        return out
