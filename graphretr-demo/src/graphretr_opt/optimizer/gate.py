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

All modes share the complexity wall: a candidate whose AST complexity exceeds
max_complexity is ineligible regardless of score. This is what makes a
sprawling memorized-keyword-table program (run1/run2's overfit mode) a hard
reject instead of a style complaint.
"""
from ..reward.pareto import dominates

DEFAULT_BLEND = {"recall@20": 0.6, "mrr": 0.4}


class Gate:
    def __init__(self, mode="strict", metric="recall@20", blend=None,
                 max_complexity=0.0):
        self.mode = mode
        self.metric = metric
        self.blend = blend or DEFAULT_BLEND
        self.max_complexity = float(max_complexity or 0.0)

    def composite(self, mv) -> float:
        return sum(w * mv.get(k) for k, w in self.blend.items())

    def accept(self, candidate, incumbent) -> bool:
        if getattr(candidate, "crashed", False):
            return False
        if (self.max_complexity
                and getattr(candidate, "code_complexity", 0.0) > self.max_complexity):
            return False
        if self.mode == "strict":
            return candidate.get(self.metric) > incumbent.get(self.metric)
        if self.mode == "blend":
            return self.composite(candidate) > self.composite(incumbent)
        if self.mode == "dominance":
            return dominates(candidate, incumbent)
        raise ValueError(f"unknown gate mode {self.mode!r}")
