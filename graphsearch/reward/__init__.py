"""graphsearch.reward -- MCQ scoring for the company-KG search target.

Owns the MCQ-specific scorer (`McqReward`), its RewardModel adapter
(`McqRewardAdapter`), and the MCQ objective keys (`qa_objectives`). Uses the
engine's reward FRAMEWORK (`graphretr_opt.reward.objectives`: MetricVector,
code_complexity) but keeps everything MCQ-specific here -- carved out of the
engine on 2026-06-25, mirroring `starksearch.reward`.
"""
from .qa_objectives import (  # noqa: F401
    MCQ_PRIMARY, MCQ_QUALITY_KEYS, mcq_dominates, mcq_objective_tuple)
from .mcq_reward import McqReward, StubAnswerer  # noqa: F401
from .mcq_reward_adapter import McqRewardAdapter  # noqa: F401
