"""env -- the IMMUTABLE half: graph backend + closed primitive DSL + sandbox.

Nothing in this package is ever mutated by the optimizer loop. It is the
SkillOpt "harness h": swapping the graph engine (backends/), the cache policy
(cache.py) or the isolation mechanism (sandbox.py) must never require touching
the optimizer/ half.

`Sandbox`/`SandboxError` are stdlib-only and exported eagerly. `RetrievalGraph`
(numpy + embedder + primitives) is exposed LAZILY so the graph_search path -- whose
eval seam is env.search_target / env.targets, NOT RetrievalGraph -- can import the
package without that chain.
"""
from .sandbox import Sandbox, SandboxError   # noqa: F401


def __getattr__(name):  # PEP 562 lazy attribute
    if name == "RetrievalGraph":
        from .retrieval_graph import RetrievalGraph
        return RetrievalGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
