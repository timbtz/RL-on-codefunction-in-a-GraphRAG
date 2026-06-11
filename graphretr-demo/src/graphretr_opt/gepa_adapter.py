"""gepa_adapter.py -- STAGE-2 SEAM. The only new file Stage 2 needs.

GEPA slots in as the optimization harness UNDER the same interfaces, because
the demo was built dict/vector-first. This adapter wraps the existing env +
reward so GEPA drives them:

  evaluate(candidate)        -> Sandbox.run + RewardModel.score, returning the
                                score AND a textual trace (retrieved-vs-gold
                                node IDs = GEPA's Actionable Side Information).
  make_reflective_dataset()  <- FastLoop._worst_failures, reshaped to GEPA's
                                reflective-dataset format.

Selection becomes GEPA's native Pareto-over-validation frontier + merge.
EditBudget / RejectedBuffer / MomentumField move from driving FastLoop to
CONSTRAINING GEPA's reflection LM (GEPA lacks all three). That is the
"fork GEPA + graft SkillOpt" decision made concrete.
"""


class GEPAAdapter:
    def __init__(self, graph, reward, substrate):
        self._graph = graph
        self._reward = reward
        self._substrate = substrate

    def evaluate(self, candidate, idxs):
        raise NotImplementedError("Stage-2 seam: wrap RewardModel.score + trace")

    def make_reflective_dataset(self, rollout_rows):
        raise NotImplementedError("Stage-2 seam: reshape worst-failures for GEPA")
