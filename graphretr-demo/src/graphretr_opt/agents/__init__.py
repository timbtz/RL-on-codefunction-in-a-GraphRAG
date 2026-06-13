"""agents -- the optimizer-model variants (the research axis).

SingleCoder: one model owns digest -> diagnose -> edit. (Default.)
TieredCoder: Haiku analyst compresses evidence, Sonnet editor proposes, Opus
    architect takes over on plateau.

Both expose the same digest(evidence) + complete(prompt, tier) -> text
interface, so swapping them is a drop-in ablation under equal token budget.
"""
from .single import SingleCoder  # noqa: F401
from .team import TieredCoder    # noqa: F401
