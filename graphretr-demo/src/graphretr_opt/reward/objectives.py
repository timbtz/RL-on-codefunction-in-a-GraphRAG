"""objectives.py -- the metric vector and its component probes.

Reward is a VECTOR, never scalarized. This is enforced from Stage 1 so that
(a) the gate can stay strict-greater on one quality axis while the candidate
pool keeps the whole vector, and (b) GEPA's Pareto frontier has a real vector to
work with in Stage 2. Scalarizing here is what produces the "retrieve
everything" degenerate, so we don't.

Axes:
  quality   -- {recall@20, hit@1, hit@5, mrr} node-containment (STaRK Evaluator)
  recall_at_100 -- large-K reachability diagnostic (Phase C2, NON-gated): when
                   gold is reachable recall@100 >> recall@20, so the gap tells us
                   per query whether the bottleneck is generation or ranking.
  latency_s -- mean wall-clock per query (lower better)
  db_load   -- mean backend queries per query (lower better)
  code_complexity -- proxy for program size/branching (lower better)
  crashed_frac, crashed -- reliability bookkeeping
  per_query -- {idx: {mrr, hit@1, recall@100}} retained per gate query (Phase A1)
               so the candidate pool can select instance-wise (sole-best on a
               single query), not just on the aggregate. NOT scalarized into the
               gate and NOT emitted to MLflow -- it is selection metadata only.
"""
import ast
from dataclasses import dataclass, field, asdict

# hit@5 added for the Hit@1 push: the leaderboard reports it and it shows rank
# movement the binary hit@1 misses. recall@20 stays the Pareto/'primary' axis.
QUALITY_KEYS = ("recall@20", "hit@1", "hit@5", "mrr")


@dataclass
class MetricVector:
    quality: dict = field(default_factory=lambda: {k: 0.0 for k in QUALITY_KEYS})
    latency_s: float = 0.0
    db_load: float = 0.0
    llm_calls: float = 0.0
    rerank_items: float = 0.0             # 0.6: mean items/query sent to llm_rerank
                                          # -- the deterministic cost meter (post-mortem
                                          # #5). NON-gated: cost-aware EXPORT re-pick only.
    code_complexity: float = 0.0
    crashed_frac: float = 0.0
    crashed: bool = False
    recall_at_100: float = 0.0            # Phase C2 diagnostic axis (non-gated)
    per_query: dict = field(default_factory=dict)  # Phase A1: idx -> {mrr, hit@1, recall@100}

    @property
    def primary(self) -> float:
        """The headline quality axis the Stage-1 gate compares (recall@20)."""
        return self.quality.get("recall@20", 0.0)

    def get(self, metric: str) -> float:
        return self.quality.get(metric, 0.0)

    def as_flat(self) -> dict:
        """Flat dict for MLflow logging (no '@' -- not a legal metric char).
        per_query is selection-only metadata and is deliberately omitted (it is
        a dict, not a scalar metric)."""
        out = {f"quality_{k.replace('@', '_at_')}": v for k, v in self.quality.items()}
        out.update(latency_s=self.latency_s, db_load=self.db_load,
                   llm_calls=self.llm_calls, rerank_items=self.rerank_items,
                   recall_at_100=self.recall_at_100,
                   code_complexity=self.code_complexity,
                   crashed_frac=self.crashed_frac, crashed=float(self.crashed))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MetricVector":
        """Rebuild from `to_dict()` output (Phase-1 checkpoint resume). per_query
        keys are JSON-stringified on the round-trip, so coerce them back to int:
        the pool's sole-best counting compares the SAME query idx across members,
        and a resumed (str-keyed) member must still match a freshly-scored
        (int-keyed) one or it would silently win/lose every shared query."""
        d = dict(d)
        pq = d.get("per_query") or {}
        d["per_query"] = {int(k): v for k, v in pq.items()}
        return cls(**d)


def code_complexity(src: str) -> float:
    """Cheap static proxy: AST node count + branch/loop/call weighting,
    normalized so a tiny seed ~ small and a sprawling program ~ large.
    Used as a Pareto axis (prefer the simpler of two equal-quality programs)
    in the dominance test the candidate pool selects on."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 1e6
    n = sum(1 for _ in ast.walk(tree))
    branches = sum(1 for x in ast.walk(tree)
                   if isinstance(x, (ast.If, ast.For, ast.While, ast.comprehension)))
    return float(n + 3 * branches)
