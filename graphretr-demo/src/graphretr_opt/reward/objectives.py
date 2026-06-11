"""objectives.py -- the metric vector and its component probes.

Reward is a VECTOR, never scalarized. This is enforced from Stage 1 so that
(a) the gate can stay strict-greater on one quality axis while the archive
keeps the whole vector, and (b) GEPA's Pareto frontier has a real vector to
work with in Stage 2. Scalarizing here is what produces the "retrieve
everything" degenerate, so we don't.

Axes:
  quality   -- {recall@20, hit@1, mrr} node-containment (STaRK Evaluator)
  latency_s -- mean wall-clock per query (lower better)
  db_load   -- mean backend queries per query (lower better)
  code_complexity -- proxy for program size/branching (lower better)
  crashed_frac, crashed -- reliability bookkeeping
"""
import ast
from dataclasses import dataclass, field, asdict

QUALITY_KEYS = ("recall@20", "hit@1", "mrr")


@dataclass
class MetricVector:
    quality: dict = field(default_factory=lambda: {k: 0.0 for k in QUALITY_KEYS})
    latency_s: float = 0.0
    db_load: float = 0.0
    code_complexity: float = 0.0
    crashed_frac: float = 0.0
    crashed: bool = False

    @property
    def primary(self) -> float:
        """The headline quality axis the Stage-1 gate compares (recall@20)."""
        return self.quality.get("recall@20", 0.0)

    def get(self, metric: str) -> float:
        return self.quality.get(metric, 0.0)

    def as_flat(self) -> dict:
        """Flat dict for MLflow logging (no '@' -- not a legal metric char)."""
        out = {f"quality_{k.replace('@', '_at_')}": v for k, v in self.quality.items()}
        out.update(latency_s=self.latency_s, db_load=self.db_load,
                   code_complexity=self.code_complexity,
                   crashed_frac=self.crashed_frac, crashed=float(self.crashed))
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def code_complexity(src: str) -> float:
    """Cheap static proxy: AST node count + branch/loop/call weighting,
    normalized so a tiny seed ~ small and a sprawling program ~ large.
    Used as a Pareto axis (prefer the simpler of two equal-quality programs)
    and available to the slow loop's MAP-Elites cells."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 1e6
    n = sum(1 for _ in ast.walk(tree))
    branches = sum(1 for x in ast.walk(tree)
                   if isinstance(x, (ast.If, ast.For, ast.While, ast.comprehension)))
    return float(n + 3 * branches)
