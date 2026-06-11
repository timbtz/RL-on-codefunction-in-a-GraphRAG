"""SlowLoop -- STAGE-2 SEAM (not used by the Stage-1 demo).

The cross-strategy controller: wraps N FastLoop arms (one per seed family),
advances them under Hyperband successive-halving, and when the active arm
plateaus asks a big-change agent to propose a NEW strategy family
(ADAS/OPRO-conditioned on the champion + runners-up). Strategy choice becomes
an experiment the optimizer runs, not a hand-pick.

Interfaces it will use (all already present):
  FastLoop.run(substrate, seed, steps, campaign) -> ArmResult
  ParetoArchive (frontier + MAP-Elites cells; keep losing-but-cheaper arms)
  ArmScheduler (Hyperband parity, hysteresis, min-dwell, tabu)
"""


class SlowLoop:
    def __init__(self, *a, **kw):
        raise NotImplementedError("Stage-2 seam: see module docstring")

    def run(self, families, substrate):
        raise NotImplementedError
