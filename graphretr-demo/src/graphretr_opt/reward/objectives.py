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
import io
import math
import tokenize
from dataclasses import dataclass, field, asdict


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass
class MetricVector:
    # The quality axis NAMES are target-specific and owned by the per-target
    # reward (STaRK: starksearch.reward.QUALITY_KEYS; MCQ: qa_objectives.
    # MCQ_QUALITY_KEYS). The framework stays agnostic: a bare vector starts with
    # an empty quality dict and `primary` reads whichever axis `primary_key`
    # names. The reward always constructs MetricVector with an explicit `quality`.
    quality: dict = field(default_factory=dict)
    latency_s: float = 0.0
    db_load: float = 0.0
    llm_calls: float = 0.0
    rerank_items: float = 0.0             # 0.6: mean items/query sent to llm_rerank
                                          # -- the deterministic cost meter (post-mortem
                                          # #5). NON-gated: cost-aware EXPORT re-pick only.
    code_complexity: float = 0.0
    code_tokens: float = 0.0              # program SIZE in source tokens (the "K"
                                          # axis of the value gate). User-chosen
                                          # complexity unit -- tokens of code, not
                                          # AST nodes -- so a doubling of source
                                          # size is a literal 2x. Lower better.
    crashed_frac: float = 0.0
    crashed: bool = False
    recall_at_100: float = 0.0            # Phase C2 diagnostic axis (non-gated)
    usd_cost: float = 0.0                 # graph_search: REAL mean USD cost/query
                                          # (OpenRouter token_usage.cost, search +
                                          # answerer). The crucial cost axis -- in
                                          # the MCQ dominance tuple and the gate
                                          # blend, so the optimizer trades dollars
                                          # against accuracy (not raw call count).
    tokens_in: float = 0.0                # mean input tokens/query (diagnostic)
    tokens_out: float = 0.0               # mean output tokens/query (diagnostic)
    per_query: dict = field(default_factory=dict)  # Phase A1: idx -> {mrr, hit@1, recall@100}
    primary_key: str = "recall@20"        # R5: which quality axis `primary` reads.
                                          # Default = the function campaign's
                                          # recall@20; the graph_search reward sets
                                          # it to "mcq_accuracy" so the same
                                          # Pareto/dominance/pool code compares the
                                          # right axis without a second pool.

    @property
    def primary(self) -> float:
        """The headline quality axis the dominance/pool code compares. Defaults
        to recall@20 (function campaign); MCQ rewards set primary_key to
        "mcq_accuracy" so dominance ranks on the MCQ axis (Risk R5)."""
        return self.quality.get(self.primary_key, 0.0)

    def get(self, metric: str) -> float:
        """Quality axes first; fall back to top-level cost attrs (usd_cost,
        llm_calls, latency_s, ...) so a cost-aware gate `blend` can weight them
        (e.g. "mcq_accuracy:1.0,usd_cost:-5.0")."""
        if metric in self.quality:
            return self.quality[metric]
        v = getattr(self, metric, None)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    def as_flat(self) -> dict:
        """Flat dict for MLflow logging (no '@' -- not a legal metric char).
        per_query is selection-only metadata and is deliberately omitted (it is
        a dict, not a scalar metric)."""
        out = {f"quality_{k.replace('@', '_at_')}": v for k, v in self.quality.items()}
        out.update(latency_s=self.latency_s, db_load=self.db_load,
                   llm_calls=self.llm_calls, rerank_items=self.rerank_items,
                   recall_at_100=self.recall_at_100,
                   code_complexity=self.code_complexity,
                   code_tokens=self.code_tokens,
                   usd_cost=self.usd_cost, tokens_in=self.tokens_in,
                   tokens_out=self.tokens_out,
                   crashed_frac=self.crashed_frac, crashed=float(self.crashed))
        return out

    def sanitize(self) -> "MetricVector":
        """Per-component NaN/inf guard (Phase 2). openEvolve filters non-finite
        metrics (`metrics_utils.py:24-32`) only to feed a SCALARIZED combined_score;
        we apply the same finite-check PER AXIS of the vector instead -- a candidate
        whose evaluator math produces a NaN/inf must degrade to a safe-WORST value,
        never crash the reducer/pool/dominance comparison downstream. We do NOT
        scalarize: the vector + Pareto dominance stay the comparator (objectives'
        never-scalarize rule). Any non-finite axis flags the candidate as crashed,
        so the gate and dominance reject it just like a sandbox crash. In place."""
        bad = False
        for k, v in list(self.quality.items()):
            if not _is_finite_number(v):
                self.quality[k] = 0.0
                bad = True
        for attr in ("latency_s", "db_load", "llm_calls", "rerank_items",
                     "code_complexity", "code_tokens", "crashed_frac",
                     "recall_at_100", "usd_cost", "tokens_in", "tokens_out"):
            if not _is_finite_number(getattr(self, attr)):
                setattr(self, attr, 0.0)
                bad = True
        for idx, d in list(self.per_query.items()):
            for k, v in list(d.items()):
                if not _is_finite_number(v):
                    d[k] = 0.0
                    bad = True
        if bad:
            # treat a non-finite candidate exactly like a crash: zero quality so it
            # can never win the gate or dominate a real program.
            self.quality = {k: 0.0 for k in self.quality}
            self.crashed = True
        return self

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


# Token kinds that carry no program size: layout, comments, and the structural
# markers tokenize emits around them. Everything else (NAME/OP/NUMBER/STRING/
# keywords) is real source the program pays for.
_NOISE_TOKENS = frozenset((
    tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
    tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER,
))


def code_tokens(src: str) -> float:
    """Program SIZE as a count of meaningful Python source tokens (NAME, OP,
    NUMBER, STRING, keywords) -- the "K" axis the value gate penalizes. Tokens,
    not lines, so reformatting/whitespace can't game it and a literal doubling of
    code is a 2x. Comments and layout tokens don't count (free to document).
    Unparseable source returns a large sentinel so a broken candidate looks
    maximally complex (it will crash anyway). -> float (>= 1.0 for any real src)."""
    if not src:
        return 1.0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return 1e6
    n = sum(1 for t in toks
            if t.type not in _NOISE_TOKENS and (t.string or "").strip())
    return float(max(1, n))
