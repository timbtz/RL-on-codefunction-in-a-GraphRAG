"""MomentumField -- STAGE-2 SEAM (inert in the Stage-1 demo).

SkillOpt's slow/meta field: an epoch-wise summary of what edit directions have
been paying off, fed back to bias the mutator -- "momentum" across the search.
It is protected (the gate cannot evict it) and updated only at epoch
boundaries.

Stage-1 constructs it but never calls update(), so context() returns "" and it
costs nothing. Stage-2 fills update() (summarize accepted-edit trajectories
into a short directive) and threads context() into the prompt next to the
rejected buffer.
"""


class MomentumField:
    def __init__(self):
        self._summary = ""
        self.protected = True

    def update(self, epoch_trajectories) -> str:
        """Stage-2: distill accepted-edit history into a short steering note."""
        return self._summary  # inert in Stage-1

    def context(self) -> str:
        return self._summary
