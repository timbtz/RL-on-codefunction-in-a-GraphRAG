"""Gate -- the validation acceptance test.

Stage-1 mode 'strict': accept iff the candidate is non-crashing and beats the
incumbent strictly on one quality axis (recall@20). Strict-greater + a fixed
gate subsample is what absorbs gate noise (ties rejected).

Mode 'blend': accept iff a weighted composite of several quality axes (e.g.
0.6*recall@20 + 0.4*mrr) strictly improves. This is the anti-overfit gate:
rewarding recall AND ranking together stops the optimizer from pumping recall@20
in isolation (which it did in run1/run2, leaving hit@1/mrr flat and overfit).

Stage-2 mode 'dominance': accept iff the candidate Pareto-dominates the
incumbent across the metric vector. Same call site, swap the mode.

Mode 'value': accept iff the candidate's efficiency-adjusted VALUE strictly
beats the incumbent's. The value is a multiplicative (Cobb-Douglas) trade-off

    V = Q / (C^cost_exp * K^cmpl_exp),   V = 0 if crashed

with Q the quality blend, C the real $/query (floored), K the program size in
source tokens (floored). This is the parsimony form the evolutionary-search
literature endorses over an additive `Q - c*cost` penalty (which is un-tunable
and gameable -- Luke & Panait 2002; Ng et al. 1999): the penalty scales WITH
fitness, so a crash gates to 0 (KernelBench/Kevin's multiplicative `1{valid}`)
and runaway cost/complexity cannot be bought by a marginal quality bump. The
exponents are calibrated so a doubling of a resource demands a fixed % quality
gain to break even (cost x2 -> +18.75%, complexity x2 -> +6.25%), compounding
geometrically with each further doubling.

All modes share the complexity wall: a candidate whose program-token size
exceeds max_tokens (or AST complexity exceeds max_complexity) is ineligible
regardless of score -- a hard reject, not a style complaint. This is what stops
the run10c bloat spiral (complexity climbing 2716 -> 4607 unchecked).
"""
import math

from ..reward.pareto import dominates

DEFAULT_BLEND = {"recall@20": 0.6, "mrr": 0.4}


class Gate:
    def __init__(self, mode="strict", metric="recall@20", blend=None,
                 max_complexity=0.0, max_tokens=0.0,
                 cost_exp=0.0, complexity_exp=0.0,
                 cost_floor=5e-4, tokens_floor=1.0):
        self.mode = mode
        self.metric = metric
        self.blend = blend or DEFAULT_BLEND
        self.max_complexity = float(max_complexity or 0.0)
        self.max_tokens = float(max_tokens or 0.0)
        # Value-mode exponents (alpha=cost, beta=complexity) + the floors that
        # cap cheap-side credit so a degenerate ultra-cheap/tiny program can't win
        # the cost term (the research's "saturate/cap" guard).
        self.cost_exp = float(cost_exp or 0.0)
        self.complexity_exp = float(complexity_exp or 0.0)
        self.cost_floor = max(float(cost_floor or 0.0), 1e-9)
        self.tokens_floor = max(float(tokens_floor or 0.0), 1.0)

    def composite(self, mv) -> float:
        return sum(w * mv.get(k) for k, w in self.blend.items())

    def value(self, mv) -> float:
        """Efficiency-adjusted scalar V = Q / (C^a * K^b); 0 if crashed. Used to
        rank for the headline-best / export decision (selection scalar), while the
        Pareto pool keeps the full vector for diversity."""
        if getattr(mv, "crashed", False):
            return 0.0
        q = self.composite(mv)
        if q <= 0.0:
            return 0.0
        c = max(float(getattr(mv, "usd_cost", 0.0)), self.cost_floor)
        k = max(float(getattr(mv, "code_tokens", 0.0)), self.tokens_floor)
        penalty = math.exp(self.cost_exp * math.log(c)
                           + self.complexity_exp * math.log(k))
        return q / penalty if penalty > 0.0 else 0.0

    def eligible(self, candidate) -> bool:
        """The shared hard wall: crashes and over-cap programs are ineligible in
        every mode, regardless of score."""
        if getattr(candidate, "crashed", False):
            return False
        if (self.max_tokens
                and getattr(candidate, "code_tokens", 0.0) > self.max_tokens):
            return False
        if (self.max_complexity
                and getattr(candidate, "code_complexity", 0.0) > self.max_complexity):
            return False
        return True

    def accept(self, candidate, incumbent) -> bool:
        if not self.eligible(candidate):
            return False
        if self.mode == "strict":
            return candidate.get(self.metric) > incumbent.get(self.metric)
        if self.mode == "blend":
            return self.composite(candidate) > self.composite(incumbent)
        if self.mode == "value":
            return self.value(candidate) > self.value(incumbent)
        if self.mode == "dominance":
            return dominates(candidate, incumbent)
        raise ValueError(f"unknown gate mode {self.mode!r}")
