"""NullSandbox -- the no-op sandbox for the graph_search target.

On the function campaign, Sandbox AST-gates, compiles and runs a `def search(q,
G)` in-process. On the graph_search campaign there is NO in-process execution:
candidates are whole real files run in an isolated subprocess by the
SearchTarget, so the in-process sandbox collapses to a pass-through.

`compile(x)` returns the artifact unchanged (FastLoop's `_get_fn` does
`compile(p.src)`, and FileSet.src is the FileSet itself), and `probe(...)` is a
no-op (the real "does it even import/run" check happens in the subprocess on the
gate set, not here). This satisfies FastLoop's `self._sandbox.compile/probe`
contract without an AST jail -- the deletion of the AST gate on this path, made
concrete.
"""


class NullSandbox:
    def compile(self, artifact):
        return artifact

    def probe(self, fn, probe_queries=None, timeout_s=None, min_ok=1):
        return None


class NullGraph:
    """Stub for FastLoop's `graph` on the graph_search path. The loop only reads
    `graph.get_text(ids)` (in the reflection-failure record) and `graph.describe()`
    (the function mutator's primitives doc); the graph_search reward carries its
    own context, so both are inert here."""

    def get_text(self, ids):
        return {}

    def describe(self):
        return ""

