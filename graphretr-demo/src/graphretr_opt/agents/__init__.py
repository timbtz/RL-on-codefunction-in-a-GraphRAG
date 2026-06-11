"""agents -- the optimizer-model variants (the research axis).

SingleCoder: one reflective coder owns diagnose -> edit. (Stage-1 default.)
OptimizerTeam: proposer / critic / verifier or island-model coders. (Stage-2.)

Both expose the same complete(prompt) -> text interface, so swapping them is a
drop-in ablation under equal token budget.
"""
from .single import SingleCoder  # noqa: F401
