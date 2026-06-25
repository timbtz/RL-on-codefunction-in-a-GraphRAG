"""_worker.py -- the subprocess entrypoint for SubprocessSearchTarget.

Reads ONE job as JSON from stdin:
  {
    "overlay_dir": "<throwaway copy of graphsearch/src, overlay already applied>",
    "queries":     ["...", ...],          # ONLY question strings (eval hygiene)
    "neo4j_cfg":   {url, username, password, database},
    "llm_cfg":     {provider, model},
    "service_relpath": "<optional>",
    "service_kwargs":  {optional}
  }

Puts overlay_dir on sys.path FIRST (so the *edited* candidate file is the one
imported -- a fresh process per candidate means no module caching across
candidates), imports common.service.qa_eval.qa_runner, builds `query_concurrency`
isolated (service, CostSink) slots, runs the queries across them (in parallel
when query_concurrency > 1, each query metered on its own slot), and prints:
  {"results": {query: {"context": str, "cost": {...}, "error": str|None}}}
A fatal (bad job, import/build failure) prints {"error": ..., "trace": ...}.

This is the kernel-evo `validate.py` shape: payload in -> metrics/result dict
out, domain logic only, no security sandbox (isolation is the process boundary +
the throwaway dir + the parent's wall-clock kill).
"""
import json
import queue
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor


def _one(service, sink, query, run_query):
    """Run a single query on an EXCLUSIVELY-held (service, sink) slot: reset the
    slot's sink, run, snapshot. Because the slot is owned by one thread for the
    whole call, the per-query cost metering never crosses queries."""
    sink.reset()
    entry = {"context": "", "cost": sink.snapshot(), "error": None}
    try:
        entry["context"] = run_query(service, query)
    except Exception as e:  # per-query crash -> scored as a miss, not fatal
        entry["error"] = f"{type(e).__name__}: {e}"
    entry["cost"] = sink.snapshot()
    return entry


def _run(job):
    overlay_dir = job["overlay_dir"]
    sys.path.insert(0, overlay_dir)
    # Imported only after the overlay is on the path so the candidate's edited
    # source is what loads.
    from common.service.qa_eval.qa_runner import (
        CostSink, build_service, run_query)

    queries = list(job["queries"])
    conc = max(1, int(job.get("query_concurrency", 1) or 1))
    conc = min(conc, len(queries)) if queries else 1

    def _build_slot():
        sink = CostSink()
        service = build_service(
            job["neo4j_cfg"], job["llm_cfg"], instrument=sink,
            service_relpath=job.get("service_relpath"),
            service_kwargs=job.get("service_kwargs"))
        return service, sink

    if conc <= 1:
        # Sequential: one slot reused across queries (original behaviour).
        service, sink = _build_slot()
        return {"results": {q: _one(service, sink, q, run_query) for q in queries}}

    # Parallel: a pool of `conc` ISOLATED (service, sink) slots, leased one per
    # worker thread via a thread-safe queue. Each slot has its own CostSink and
    # its own metered chat model, so concurrent queries never share metering.
    # The searches themselves are read-only over Neo4j -> safe to run at once.
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
