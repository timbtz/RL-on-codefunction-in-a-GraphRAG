"""STaRK reward: node-containment recall@20/hit@1/hit@5/mrr scoring.

Uses the engine's reward framework (`graphretr_opt.reward.objectives`:
MetricVector, code_complexity) but owns the STaRK-specific `QUALITY_KEYS` and
the `RewardModel` that runs a candidate over queries and scores it.
"""
from .evaluator import QUALITY_KEYS, RewardModel  # noqa: F401
from .subprocess_reward import StarkRewardAdapter  # noqa: F401
