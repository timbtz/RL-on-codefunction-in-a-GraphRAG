"""search_target.py -- the deep eval seam for the `graph_search` target.

This is the port that replaces the in-process Sandbox+RetrievalGraph on this
path. A tiny interface ("given candidate files + queries, return retrieved
context + a cost meter") hides a large implementation (repo overlay, subprocess
spawn, the real LangChain agentic service, Neo4j). High leverage: one interface
scores any candidate FileSet. Strong locality: every "how we run a candidate"
detail lives behind it, in an adapter.

Two adapters justify the port (one adapter = a hypothetical seam, two = a real
one): SubprocessSearchTarget (production -- env/targets/subprocess_target.py) and
FakeSearchTarget (test -- env/targets/fake_target.py). The fake makes the whole
optimizer loop runnable with no Neo4j and no API keys: the interface IS the test
surface.

No Neo4j / LangChain types appear in this file -- adapters own those. The port
speaks only `FileSet`, plain strings, and CostMeter.
"""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class CostMeter:
    """Per-run cost accounting for one candidate over a batch of queries -- the
    process-level analogue of sandbox.RunStats. Aggregated across queries by the
    adapter; the reward turns it into the MetricVector cost axes (latency_s /
    llm_calls / db_load)."""
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    db_queries: int = 0
    latency_s: float = 0.0
    usd_cost: float = 0.0          # REAL OpenRouter USD cost over this run

    def merge(self, other: "CostMeter") -> "CostMeter":
        """Sum two meters (batch aggregation). Returns a new meter."""
        return CostMeter(
            llm_calls=self.llm_calls + other.llm_calls,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            db_queries=self.db_queries + other.db_queries,
            latency_s=self.latency_s + other.latency_s,
            usd_cost=self.usd_cost + other.usd_cost,
        )

    def to_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "db_queries": self.db_queries,
                "latency_s": self.latency_s, "usd_cost": self.usd_cost}

    @classmethod
    def from_dict(cls, d: dict) -> "CostMeter":
        return cls(**{k: d.get(k, 0) for k in
                      ("llm_calls", "tokens_in", "tokens_out", "db_queries")},
                   latency_s=float(d.get("latency_s", 0.0)),
                   usd_cost=float(d.get("usd_cost", 0.0)))


@dataclass
class SearchResult:
    """What an adapter returns for ONE query: the retrieved context, its cost,
    and an optional error string (set when the candidate crashed/hung on this
    query -- the reward scores that as a miss, not a fatal)."""
    context: str
    cost: CostMeter = field(default_factory=CostMeter)
    error: str | None = None


@runtime_checkable
class SearchTarget(Protocol):
    """The eval seam. Batch by design so a subprocess amortizes startup across
    the whole gate set.

    run(file_set, queries, timeout_s) -> {query: SearchResult}

    Only the QUESTION strings cross this seam -- never the MCQ choices or the
    gold answer index (eval hygiene: the search must not see the key)."""

    def run(self, file_set, queries, timeout_s: float) -> dict:
        ...
