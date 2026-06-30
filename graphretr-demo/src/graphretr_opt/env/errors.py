"""env/errors.py -- engine-stable exception + builtin allowlist.

`SandboxError` is the candidate-rejection signal the optimizer core catches in
its rollout loop (`fast_loop`, `mutator`) and the per-target reward raises when a
candidate crashes / hangs / returns garbage. It is intentionally split out of
`sandbox.py` so it survives the loosening of the in-process AST sandbox: the core
keeps catching one stable exception type while the actual isolation mechanism
(in-process gate vs subprocess `FileSet` overlay) varies per target.

`SAFE_BUILTINS` is the restricted builtin namespace the in-process STaRK sandbox
exec's candidates into, and the allowlist the mutator advertises to the LLM. It
lives here (stdlib-only) so importers don't drag in the AST gate.
"""
import builtins

_SAFE_BUILTIN_NAMES = (
    "len", "sorted", "set", "dict", "list", "tuple", "min", "max", "sum",
    "enumerate", "zip", "range", "abs", "round", "float", "int", "str",
    "bool", "any", "all", "reversed",
)
SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}


class SandboxError(Exception):
    pass
