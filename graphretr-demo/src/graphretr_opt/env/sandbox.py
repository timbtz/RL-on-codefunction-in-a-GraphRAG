"""sandbox.py -- AST safety gate + compile + isolated execution of a candidate
program, returning (pred, RunStats).

Hard walls, not discipline:
  * AST gate: no imports, no dunder/underscore attribute access, no names
    outside {q, G, locally-defined} + a small builtin allowlist, no
    class/async/with/global. The module must define `def search(q, G)`.
  * Compile: exec'd into a namespace whose __builtins__ contains ONLY the
    allowlist -- even if the AST gate were bypassed there is nothing to grab.
  * Execution: a SIGALRM wall-clock kill bounds every call. (Stage-2 seam:
    swap _run_once for a warm worker pool with setrlimit + no-net; the
    Sandbox.run signature stays identical.)

RunStats carries wall-clock latency and backend query count so the reward's
latency / db-load objectives have real numbers without a second harness.
"""
import ast
import builtins
import signal
import time
from dataclasses import dataclass

_SAFE_BUILTIN_NAMES = (
    "len", "sorted", "set", "dict", "list", "tuple", "min", "max", "sum",
    "enumerate", "zip", "range", "abs", "round", "float", "int", "str",
    "bool", "any", "all", "reversed",
)
SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}

_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "type", "object", "super",
    "input", "breakpoint", "memoryview", "bytearray", "bytes", "print",
}

_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.With, ast.AsyncWith,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.Await, ast.Global, ast.Nonlocal,
)


class SandboxError(Exception):
    pass


class _Timeout(Exception):
    pass


@dataclass
class RunStats:
    latency_s: float
    queries: int
    llm_calls: int = 0
    rerank_items: int = 0   # 0.6 cost meter: items the program sent to llm_rerank


class time_limit:
    """SIGALRM-based wall-clock limit (main thread only -- which we are)."""

    def __init__(self, seconds):
        self.seconds = float(seconds)

    def __enter__(self):
        def handler(signum, frame):
            raise _Timeout()
        self._old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old)
        return False


def _defined_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def check_source(src):
    """Raise SandboxError unless src is an acceptable candidate module."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}")

    search_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "search":
                search_def = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            pass  # module-level constants are fine
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass  # docstring
        else:
            raise SandboxError(
                f"top-level {type(node).__name__} not allowed (only `def`, "
                "constant assignments and a docstring)")
    if search_def is None:
        raise SandboxError("module must define `def search(q, G)`")
    a = search_def.args
    if ([x.arg for x in a.args] != ["q", "G"] or a.vararg or a.kwarg
            or a.kwonlyargs or a.posonlyargs or a.defaults):
        raise SandboxError("search signature must be exactly `def search(q, G)`")

    allowed = set(_SAFE_BUILTIN_NAMES) | _defined_names(tree) | {"q", "G"}
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxError(f"underscore attribute access `.{node.attr}` is not allowed")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _FORBIDDEN_NAMES:
                raise SandboxError(f"`{node.id}` is not allowed")
            if node.id not in allowed:
                raise SandboxError(
                    f"unknown name `{node.id}` (allowed: q, G, your own "
                    f"definitions, and builtins {sorted(_SAFE_BUILTIN_NAMES)})")
    return tree


def compile_program(src):
    """AST-gate then exec src in an empty-builtins namespace -> search callable."""
    check_source(src)
    ns = {"__builtins__": dict(SAFE_BUILTINS)}
    try:
        exec(compile(src, "<candidate>", "exec"), ns)
    except Exception as e:
        raise SandboxError(f"module exec failed: {type(e).__name__}: {e}")
    fn = ns.get("search")
    if not callable(fn):
        raise SandboxError("`search` is not callable after exec")
    return fn


def validate_pred(pred):
    """Normalize a candidate's return value to dict[int, float] or raise."""
    if not isinstance(pred, dict):
        raise SandboxError(
            f"search() must return dict[int, float], got {type(pred).__name__}")
    if not pred:
        raise SandboxError("search() returned an empty dict")
    out = {}
    for k, v in pred.items():
        if isinstance(k, bool) or not isinstance(k, int):
            raise SandboxError(f"pred key {k!r} is not an int node id")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SandboxError(f"pred value {v!r} for id {k} is not a number")
        out[k] = float(v)
    return out


class Sandbox:
    """Isolated executor for candidate programs. Compile once, run many."""

    def __init__(self, graph, default_timeout_s=10.0):
        self._graph = graph
        self._default_timeout_s = default_timeout_s

    def compile(self, src):
        return compile_program(src)

    def run(self, fn, query, timeout_s=None):
        """Execute a compiled candidate on one query.
        -> (pred: dict[int, float], stats: RunStats). Raises SandboxError on
        crash, hang, or malformed return."""
        timeout_s = self._default_timeout_s if timeout_s is None else timeout_s
        backend = getattr(self._graph, "_backend", None)
        q0 = getattr(backend, "query_count", 0)
        l0 = getattr(self._graph, "_llm_calls", 0)
        ri0 = getattr(self._graph, "_rerank_items", 0)
        pin = getattr(self._graph, "_pin_query", None)
        if pin:
            pin(query)
        t0 = time.time()
        try:
            with time_limit(timeout_s):
                pred = fn(query, self._graph)
        except _Timeout:
            raise SandboxError(f"timed out (> {timeout_s}s) on: {query[:120]!r}")
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(f"crashed on {query[:120]!r}: {type(e).__name__}: {e}")
        pred = validate_pred(pred)
        stats = RunStats(latency_s=time.time() - t0,
                         queries=getattr(backend, "query_count", q0) - q0,
                         llm_calls=getattr(self._graph, "_llm_calls", l0) - l0,
                         rerank_items=getattr(self._graph, "_rerank_items", ri0) - ri0)
        return pred, stats

    def probe(self, fn, probe_queries, timeout_s=None, min_ok=1):
        """Tolerant probe cascade (post-mortem #3 / Phase 0.4): run EVERY fixed
        probe query and require at least `min_ok` of them to return a valid
        non-empty pred. A legitimately-empty intermediate on a hard query (e.g.
        the fixed MAPK-kinase probe `ids is empty`) is a per-query MISS, not a
        programmer error -- the authoritative gate (evaluator.py) already scores
        per-query and only fails a candidate when crashed_frac exceeds the limit,
        so a strict all-must-pass probe was a redundant, stricter inconsistency
        that hard-crashed otherwise-scorable candidates *after* paying the
        mutator (run-7: 483K wasted tokens, 3 of 15 Opus calls, zero signal) and
        fed crash-driven premature stops. Re-raise only when fewer than `min_ok`
        queries succeed (a genuinely broken program). -> the last good pred (or
        None if probe_queries is empty)."""
        if not probe_queries:
            return None
        last, ok, errors = None, 0, []
        for query in probe_queries:
            try:
                pred, _ = self.run(fn, query, timeout_s)
                last, ok = pred, ok + 1
            except SandboxError as e:
                errors.append(str(e))
        if ok < min_ok:
            n = len(probe_queries)
            raise SandboxError(
                f"probe failed: only {ok}/{n} probe queries returned a valid "
                f"pred (need >= {min_ok}); first errors: {errors[:3]}")
        return last
