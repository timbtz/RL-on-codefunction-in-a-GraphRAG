"""STaRK reward: node-containment recall@20/hit@1/hit@5/mrr scoring.

Uses the engine's reward framework (`graphretr_opt.reward.objectives`:
MetricVector, code_complexity) but owns the STaRK-specific `QUALITY_KEYS` and the
`StarkRewardAdapter` that runs a candidate FileSet through the subprocess STaRK
target and scores each returned pred (the in-process `RewardModel`/AST sandbox
was deleted with the carve-out -- the subprocess adapter is the single scorer).
"""
from .evaluator import QUALITY_KEYS  # noqa: F401
from .subprocess_reward import StarkRewardAdapter  # noqa: F401
