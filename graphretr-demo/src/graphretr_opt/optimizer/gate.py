"""Gate -- the validation acceptance test.

Stage-1 mode 'strict': accept iff the candidate is non-crashing and beats the
incumbent strictly on one quality axis (recall@20). Strict-greater + a fixed
gate subsample is what absorbs gate noise (ties rejected).

Stage-2 mode 'dominance': accept iff the candidate Pareto-dominates the
incumbent across the metric vector. Same call site, swap the mode.
"""
from ..reward.pareto import dominates


class Gate:
    def __init__(self, mode="strict", metric="recall@20"):
        self.mode = mode
        self.metric = metric

    def accept(self, candidate, incumbent) -> bool:
        if getattr(candidate, "crashed", False):
            return False
        if self.mode == "strict":
            return candidate.get(self.metric) > incumbent.get(self.metric)
        if self.mode == "dominance":
            return dominates(candidate, incumbent)
        raise ValueError(f"unknown gate mode {self.mode!r}")
