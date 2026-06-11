"""EditBudget -- the per-step diff-hunk budget L_t. SkillOpt's analogue of a
learning rate: how many regions the mutator may change this step.

const (Stage-1 default): L_t = L_max every step.
cosine / linear: decay from L_max to L_min over the campaign -- big structural
edits early, fine-tuning late.
"""
import math


class EditBudget:
    def __init__(self, schedule="const", l_max=4, l_min=1, steps=30):
        self.schedule = schedule
        self.l_max = l_max
        self.l_min = l_min
        self.steps = max(1, steps)

    def L_t(self, step: int) -> int:
        if self.schedule == "const":
            return self.l_max
        frac = min(1.0, step / self.steps)
        if self.schedule == "linear":
            val = self.l_max - (self.l_max - self.l_min) * frac
        elif self.schedule == "cosine":
            val = self.l_min + 0.5 * (self.l_max - self.l_min) * (1 + math.cos(math.pi * frac))
        else:
            raise ValueError(f"unknown edit schedule {self.schedule!r}")
        return max(self.l_min, int(round(val)))
