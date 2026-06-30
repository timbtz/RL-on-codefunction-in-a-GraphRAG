"""env -- the engine's target-agnostic eval seam + shared gateways.

Nothing here is mutated by the optimizer loop. After the STaRK carve-out
(2026-06-25) the STaRK `G` service (retrieval_graph, primitives, cache,
embedder, backends) lives in the sibling `starksearch/` package, and BOTH
targets now score candidates in a worker subprocess -- so the in-process AST
`Sandbox` is gone. What remains is the generic seam: the `SearchTarget` port
(`search_target`, `targets/`: the graphsearch + STaRK workers), the shared
LLM/budget gateway (`openai_client`), the no-op `null_sandbox`, and the stable
`SandboxError` (`errors`) the optimizer core still catches.
"""
from .errors import SandboxError   # noqa: F401
