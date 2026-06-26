"""_worker_stark.py -- the subprocess entrypoint for the STaRK search target.

The STaRK analogue of `_worker.py`. Reads ONE job as JSON from stdin:
  {
    "overlay_dir":       "<throwaway copy of starksearch/src, overlay applied>",
    "queries":           ["...", ...],          # ONLY question strings (eval hygiene)
    "falkor_cfg":        {host, port, graph_name},
    "cfg_overrides":     {root, ...},            # forwarded to load_config()
    "query_concurrency": <int>
  }

Puts overlay_dir on sys.path FIRST (so the EDITED service file is the one that
loads -- a fresh process per candidate means no module caching across
candidates), imports `stark_harness.qa_runner` from the overlay, builds
`query_concurrency` isolated (service, CostSink) slots, runs the queries across
them (in parallel when query_concurrency > 1, each metered on its own slot), and
prints:
  {"results": {query: {"pred": {node_id: score}, "cost": {...}, "error": str|None}}}
A fatal (bad job, import/build failure) prints {"error": ..., "trace": ...}.

This is the loosening of the old AST sandbox: the service is ordinary Python
(real builtins, free imports) run in a real subprocess -- isolation is the
process boundary + the parent's wall-clock kill, NOT an in-process gate. The caps
that protect the SHARED FalkorDB container (read-only backend, row caps, the
SPpaths maxLen<=3 wall) live in the harness firewall (`stark_harness.qa_runner`),
the one chokepoint every candidate query passes through.
"""
import json
import queue
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def _validate_pred(pred):
    """Normalize a candidate's return value to dict[int, float] or raise -- the
    contract the STaRK reward scores against."""
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


def _one(service, sink, query, run_query):
    """Run ONE query on an EXCLUSIVELY-held (service, sink) slot: reset the slot's
    sink, run, snapshot the per-query cost. The slot is owned by one thread for
    the whole call, so the per-query cost metering never crosses queries -- the
    isolation the in-process Sandbox.run getattr-deltas used to give."""
    sink.reset()
    entry = {"pred": {}, "cost": {}, "error": None}
    t0 = time.time()
    try:
        pred = run_query(service, query)
        entry["pred"] = {str(k): v for k, v in _validate_pred(pred).items()}
    except Exception as e:  # per-query crash -> scored as a miss, not fatal
        entry["error"] = f"{type(e).__name__}: {e}"
    cost = sink.snapshot()
    cost["latency_s"] = time.time() - t0
    entry["cost"] = cost
    return entry


def _run(job):
    overlay_dir = job["overlay_dir"]
    sys.path.insert(0, overlay_dir)
    # Imported only after the overlay is on the path so the candidate's edited
    # service is what loads.
    from stark_harness.qa_runner import CostSink, build_service, run_query

    queries = list(job["queries"])
    conc = max(1, int(job.get("query_concurrency", 1) or 1))
    conc = min(conc, len(queries)) if queries else 1

    def _build_slot():
        sink = CostSink()
        service = build_service(
            job["falkor_cfg"], job.get("cfg_overrides") or {}, instrument=sink)
        return service, sink

    if conc <= 1:
        # Sequential: one slot reused across queries.
        service, sink = _build_slot()
        return {"results": {q: _one(service, sink, q, run_query) for q in queries}}

    # Parallel: a pool of `conc` ISOLATED (service, sink) slots, leased one per
    # worker thread via a thread-safe queue. Each slot has its own CostSink so
    # concurrent queries never share metering; the searches are read-only over
    # FalkorDB -> safe to run at once.
    slots = queue.Queue()
    for _ in range(conc):
        slots.put(_build_slot())

    def _work(query):
        service, sink = slots.get()
        try:
            return query, _one(service, sink, query, run_query)
        finally:
            slots.put((service, sink))

    results = {}
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for query, entry in ex.map(_work, queries):
            results[query] = entry
    return {"results": results}


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
