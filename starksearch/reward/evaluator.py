"""STaRK quality axes.

The in-process `RewardModel` (an AST-sandbox `self._sandbox.run` scorer) was
DELETED with the carve-out: STaRK candidates are now whole `FileSet` services
scored in a subprocess by `StarkRewardAdapter` (subprocess_reward.py), the single
scorer. Only the STaRK-specific quality-axis names remain here -- the axes the
adapter and the STaRK campaign entrypoints report.

recall@20 is the Pareto/'primary' axis; hit@5 is reported for rank movement the
binary hit@1 misses. Re-exported via `starksearch.reward.__init__`.
"""

# STaRK node-containment quality axes. STaRK-specific, so it lives WITH the
# scorer (the carve-out) -- the engine's reward framework keeps only the generic
# MetricVector + primary_key-driven dominance, not this key list.
QUALITY_KEYS = ("recall@20", "hit@1", "hit@5", "mrr")
