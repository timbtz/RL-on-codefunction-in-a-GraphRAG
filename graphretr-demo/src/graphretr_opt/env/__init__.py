"""env -- the IMMUTABLE half: graph backend + closed primitive DSL + sandbox.

Nothing in this package is ever mutated by the optimizer loop. It is the
SkillOpt "harness h": swapping the graph engine (backends/), the cache policy
(cache.py) or the isolation mechanism (sandbox.py) must never require touching
the optimizer/ half.
"""
from .retrieval_graph import RetrievalGraph  # noqa: F401
from .sandbox import Sandbox, SandboxError   # noqa: F401
