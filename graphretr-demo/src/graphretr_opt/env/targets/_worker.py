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
candidates), imports common.service.qa_eval.qa_runner, builds the service once
with a CostSink, runs each query metering per-query, and prints to stdout:
  {"results": {query: {"context": str, "cost": {...}, "error": str|None}}}
A fatal (bad job, import/build failure) prints {"error": ..., "trace": ...}.

This is the kernel-evo `validate.py` shape: payload in -> metrics/result dict
out, domain logic only, no security sandbox (isolation is the process boundary +
the throwaway dir + the parent's wall-clock kill).
"""
import json
import sys
import traceback


def _run(job):
    overlay_dir = job["overlay_dir"]
    sys.path.insert(0, overlay_dir)
    # Imported only after the overlay is on the path so the candidate's edited
    # source is what loads.
    from common.service.qa_eval.qa_runner import (
        CostSink, build_service, run_query)

    sink = CostSink()
    service = build_service(
        job["neo4j_cfg"], job["llm_cfg"], instrument=sink,
        service_relpath=job.get("service_relpath"),
        service_kwargs=job.get("service_kwargs"))

    results = {}
    for query in job["queries"]:
        sink.reset()
        entry = {"context": "", "cost": sink.snapshot(), "error": None}
        try:
            entry["context"] = run_query(service, query)
        except Exception as e:  # per-query crash -> scored as a miss, not fatal
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["cost"] = sink.snapshot()
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
