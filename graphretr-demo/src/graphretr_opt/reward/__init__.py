"""reward -- the engine's scoring FRAMEWORK (target-agnostic).

`MetricVector` / `code_complexity` (pure ast + math) and `pareto.dominates` are
the generic pieces every target reuses. The per-target SCORERS live with their
service: STaRK node-containment in `starksearch.reward` (torch), MCQ accuracy in
`graphsearch.reward` (the carve-out, 2026-06-25). Nothing target-specific is
imported here, so this package stays importable without torch / stark_qa.
"""
from .objectives import MetricVector, code_complexity  # noqa: F401


def __getattr__(name):  # PEP 562 lazy attribute
    if name == "dominates":
        from .pareto import dominates
        return dominates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
