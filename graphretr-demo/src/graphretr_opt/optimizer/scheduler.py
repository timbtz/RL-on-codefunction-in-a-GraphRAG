"""ArmScheduler -- STAGE-2 SEAM (not used by the Stage-1 demo).

Allocates budget across strategy arms with Hyperband warm-start parity (equal
B0 per arm before any comparison), and decides when to switch the active arm:
a challenger must beat the incumbent by a hysteresis margin, after a min-dwell,
and outside a tabu cooldown -- so the slow loop doesn't thrash between basins.
"""


class ArmScheduler:
    def __init__(self, b0=10, hysteresis=0.01, min_dwell=5, tabu_steps=10):
        self.b0 = b0
        self.hysteresis = hysteresis
        self.min_dwell = min_dwell
        self.tabu_steps = tabu_steps

    def should_switch(self, incumbent_best, challenger_best, dwell) -> bool:
        raise NotImplementedError("Stage-2 seam: see module docstring")
