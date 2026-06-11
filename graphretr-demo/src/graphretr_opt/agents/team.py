"""OptimizerTeam -- STAGE-2 SEAM (not used by the Stage-1 demo).

The multi-agent arm of the publishable single-vs-team ablation: a
proposer/critic/verifier pipeline (or island-model parallel coders with
migration) behind the SAME complete(prompt) -> text interface SingleCoder
exposes, so it drops into Mutator unchanged and is compared under an equal
token budget.
"""
from .single import SingleCoder


class OptimizerTeam:
    def __init__(self, backend="cli", model="opus", timeout_s=900):
        # Stage-2: instantiate proposer/critic/verifier (or N island coders).
        self._proposer = SingleCoder(backend, model, timeout_s)

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "Stage-2 seam: proposer -> critic -> verifier round. "
            "Implement here; FastLoop/Mutator need no changes.")
