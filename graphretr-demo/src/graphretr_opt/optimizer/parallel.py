"""parallel.py -- the process-pool SHELL ported from openEvolve, minus its debt.

This is the Stage-2 seam the sandbox docstring marks ("swap _run_once for a warm
worker pool ... the Sandbox.run signature stays identical"). It ports only the
parts of openEvolve's `process_parallel.py` that actually work:

  * ProcessPoolExecutor(max_workers, initializer=..., initargs=...,
    max_tasks_per_child=N) -- N kept LOW because workers run untrusted candidate
    code (the only real reclamation of leaked threads/memory).
  * config crosses the spawn boundary as a FLAT plain dict, rebuilt in the worker
    (never pickle a live handle).
  * lazy ONE-TIME per-worker init cached in a module global (heavy objects -- the
    FalkorDB handle, embedder, graph -- built on first task, reused after).
  * busy-poll harvest so the controller coroutine stays free; results returned in
    INPUT ORDER; workers return PURE DATA and the controller is the single writer.

Deliberately NOT ported (openEvolve's parallelism theater -- see the plan's
DO-NOT-PORT list):
  * future.cancel() as a kill -- a no-op on already-running workers. graphretr
    relies on the sandbox's SIGALRM, which fires only on a worker's MAIN thread,
    so the scorer MUST run in the worker body directly, never via a thread
    executor. (`_run_task` below does exactly that.)
  * full-population re-pickle per submission -- callers ship only minimal per-task
    data.
  * abandon-but-never-kill timeout handling.

The harness is generic and unit-tested with a trivial picklable init/scorer (no
FalkorDB). Wiring it to the live retrieval graph -- an `init_fn` that boots
FalkorDB + embedder + Sandbox + RewardModel from the flat config per process --
is the caller's job and is gated OFF by default and whenever the OpenAI budget is
active (see FastLoop._parallel_scoring_enabled); that live path needs a running
FalkorDB to validate.
"""
import time
from concurrent.futures import ProcessPoolExecutor

# Per-worker module global. The initializer stows the (picklable, module-level)
# init_fn + its flat config arg + the scorer; the heavy component is built lazily
# on the first task and cached for the life of the worker.
_WS = {}


def _init_worker(init_fn, init_arg, scorer):
    _WS.clear()
    _WS["init_fn"] = init_fn
    _WS["init_arg"] = init_arg
    _WS["scorer"] = scorer
    _WS["component"] = None


def _run_task(task):
    # Lazy one-time build per worker (openEvolve's _lazy_init pattern).
    if _WS.get("component") is None:
        _WS["component"] = _WS["init_fn"](_WS["init_arg"])
    # Scorer runs HERE, on the worker's main thread -- so the candidate's SIGALRM
    # kill is in force. Do not offload it to a thread.
    return _WS["scorer"](_WS["component"], task)


def parallel_map(tasks, *, num_workers, init_fn, init_arg, scorer,
                 max_tasks_per_child=4, poll_s=0.01):
    """Run `scorer(component, task)` for every task across `num_workers` processes.

    `component` is built once per worker by `init_fn(init_arg)` (lazy + cached).
    `init_fn`, `init_arg`, and `scorer` MUST be picklable (module-level functions
    and a plain-dict arg -- the whole point of the flat-config rule). Results are
    returned in INPUT ORDER. A task that raises propagates that exception (callers
    that want best-effort should catch inside `scorer`).

    num_workers<=1 (or <=1 task) runs inline with no pool -- the exact serial path,
    so the default never pays for process machinery.
    """
    tasks = list(tasks)
    if num_workers <= 1 or len(tasks) <= 1:
        component = init_fn(init_arg)
        return [scorer(component, t) for t in tasks]

    results = [None] * len(tasks)
    kwargs = dict(max_workers=num_workers, initializer=_init_worker,
                  initargs=(init_fn, init_arg, scorer))
    try:
        ex = ProcessPoolExecutor(max_tasks_per_child=max_tasks_per_child, **kwargs)
    except TypeError:  # max_tasks_per_child is Py>=3.11 only
        ex = ProcessPoolExecutor(**kwargs)
    with ex:
        futs = {ex.submit(_run_task, t): i for i, t in enumerate(tasks)}
        pending = set(futs)
        while pending:
            done = {f for f in pending if f.done()}
            if not done:
                time.sleep(poll_s)
                continue
            for f in done:
                results[futs[f]] = f.result()
            pending -= done
    return results
