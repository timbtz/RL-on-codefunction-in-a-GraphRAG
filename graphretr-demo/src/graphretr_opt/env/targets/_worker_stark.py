"""_worker_stark.py -- the subprocess entrypoint for the STaRK search target.

The STaRK analogue of `_worker.py`. Reads ONE job as JSON from stdin:
  {
    "candidate_src":  "<the search(q, G) program source>",   # the mutated artifact
    "overlay_dir":    "<optional materialized FileSet dir>",  # forward-compat (FileSet seeds)
    "queries":        ["...", ...],                           # ONLY question strings
    "falkor_cfg":     {host, port, graph_name},
    "cfg_overrides":  {embedder, *_model, query_timeout_ms, runs_dir, ...},
    "opt_src":        "<graphretr-demo/src on PYTHONPATH>",
    "repo_root":      "<monorepo root so `starksearch` imports>"
  }

Builds the FIXED `G` service (`starksearch.graph.RetrievalGraph` over FalkorDB),
imports/execs the candidate `search`, runs each query SEQUENTIALLY on the one
shared `G` (so per-query cost metering never crosses queries -- exactly the
in-process `Sandbox.run` semantics this must reproduce), and prints:
  {"results": {query: {"pred": {node_id: score}, "cost": {...}, "error": str|None}}}
A fatal (bad job, build failure) prints {"error": ..., "trace": ...}.

This is the loosening of the old AST sandbox: the candidate is exec'd as ordinary
Python (real builtins, free imports) -- isolation is the process boundary + the
parent's wall-clock kill, NOT an in-process gate. The server-side caps that
protect the SHARED FalkorDB container (read-only backend, row caps, the
SPpaths maxLen<=3 wall) survive because they live in `starksearch.primitives`,
not in the removed gate.
"""
import json
import sys
import time
import traceback


def _validate_pred(pred):
    """Normalize a candidate's return value to dict[int, float] or raise -- the
    same contract sandbox.validate_pred enforced in-process."""
    if not isinstance(pred, dict):
        raise ValueError(f"search() must return dict[int, float], got {type(pred).__name__}")
    if not pred:
        raise ValueError("search() returned an empty dict")
    out = {}
    for k, v in pred.items():
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(f"pred key {k!r} is not an int node id")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"pred value {v!r} for id {k} is not a number")
        out[int(k)] = float(v)
    return out


def _compile_candidate(src, overlay_dir=None):
    """Exec the candidate source into a fresh namespace with REAL builtins (the
    loosening) and return its `search` callable. overlay_dir, when given, is put
    on sys.path first so a multi-file FileSet candidate's own imports resolve."""
    if overlay_dir and overlay_dir not in sys.path:
        sys.path.insert(0, overlay_dir)
    ns = {"__name__": "__candidate__", "__builtins__": __import__("builtins")}
    exec(compile(src, "<candidate>", "exec"), ns)
    fn = ns.get("search")
    if not callable(fn):
        raise ValueError("`search` is not callable after exec")
    return fn


def _build_graph(job):
    """Reconstruct the FIXED `G` service from the job: cfg (load_config +
    overrides), the metered OpenAI/OpenRouter budget, the FalkorDB backend, the
    embedder, and the RetrievalGraph. Mirrors campaign._boot_function exactly."""
    import os
    # opt_src -> graphretr_opt importable; repo_root -> starksearch importable.
    for p in (job.get("opt_src"), job.get("repo_root")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from graphretr_opt.config import load_config
    from graphretr_opt.env.openai_client import OpenAIBudget
    from starksearch.backends.falkordb import FalkorDBBackend
    from starksearch.cache import PrimitiveCache
    from starksearch.embedder import make_embedder
    from starksearch.graph import RetrievalGraph

    cfg = load_config(**(job.get("cfg_overrides") or {}))
    fc = job["falkor_cfg"]
    backend = FalkorDBBackend(fc["host"], fc["port"], fc["graph_name"])
    budget = None
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        budget = OpenAIBudget(os.path.join(cfg.runs_dir, "openai_usage.json"),
                              ceiling_usd=cfg.openai_budget_usd)
    graph = RetrievalGraph(cfg, backend, PrimitiveCache(),
                           make_embedder(cfg, budget), llm_budget=budget)
    return graph, backend


def _one(graph, backend, search, query):
    """Run ONE query on the shared G, snapshotting the per-query cost deltas the
    in-process RunStats carried (latency, backend queries, llm_calls, rerank
    items). A per-query crash is scored as a miss, not a fatal."""
    q0 = getattr(backend, "query_count", 0)
    l0 = getattr(graph, "_llm_calls", 0)
    ri0 = getattr(graph, "_rerank_items", 0)
    pin = getattr(graph, "_pin_query", None)
    if pin:
        pin(query)  # trusted-caller hook: the only text extract() accepts
    entry = {"pred": {}, "cost": {}, "error": None}
    t0 = time.time()
    try:
        pred = search(query, graph)
        entry["pred"] = {str(k): v for k, v in _validate_pred(pred).items()}
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
    entry["cost"] = {
        "latency_s": time.time() - t0,
        "db_queries": getattr(backend, "query_count", q0) - q0,
        "llm_calls": getattr(graph, "_llm_calls", l0) - l0,
        "rerank_items": getattr(graph, "_rerank_items", ri0) - ri0,
    }
    return entry


def _run(job):
    graph, backend = _build_graph(job)
    search = _compile_candidate(job["candidate_src"], job.get("overlay_dir"))
    queries = list(job["queries"])
    # Sequential by design: one shared G, query-by-query -- matches the
    # in-process Sandbox.run semantics the parity check asserts.
    return {"results": {q: _one(graph, backend, search, q) for q in queries}}


def main():
    raw = sys.stdin.read()
    try:
        job = json.loads(raw)
        out = _run(job)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
