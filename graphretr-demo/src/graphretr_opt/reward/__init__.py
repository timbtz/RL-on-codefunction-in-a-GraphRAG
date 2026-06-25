"""reward -- the scoring layer.

`MetricVector` / `code_complexity` are pure (ast + math) and exported eagerly.
`RewardModel` pulls torch (STaRK node-containment scoring), so it is exposed
LAZILY -- the graph_search path scores via reward.mcq_reward.McqReward (no torch)
and must be able to import the package without dragging torch in.
"""
from .objectives import MetricVector, code_complexity  # noqa: F401


def __getattr__(name):  # PEP 562 lazy attribute
    if name == "RewardModel":
        from .evaluator import RewardModel
        return RewardModel
    if name == "dominates":
        from .pareto import dominates
        return dominates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
