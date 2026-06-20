"""Phase 2 foundation: vector-wise NaN guard + the process-pool harness mechanics
+ the budget-guard that keeps parallel scoring OFF when it would be unsafe.

What is and isn't covered: the harness MECHANICS (flat config crossing the spawn
boundary, lazy one-time per-worker init, input-order results) are tested here with
a trivial picklable init/scorer -- no FalkorDB. The LIVE wiring (a worker that
boots the retrieval graph per process) needs a running FalkorDB and is gated off by
default; this suite asserts that guard, not the live path.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_parallel_foundation.py
"""
import math
import os
import tempfile

from graphretr_opt.optimizer.parallel import parallel_map
from graphretr_opt.reward.objectives import MetricVector
from graphretr_opt.reward.pareto import dominates


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


# ------------------------------------------------------------------- NaN guard

def test_sanitize_degrades_nonfinite_to_safe_worst():
    mv = MetricVector(quality={"recall@20": float("nan"), "hit@1": 0.5,
                               "hit@5": 0.5, "mrr": float("inf")},
                      latency_s=float("inf"), code_complexity=10.0,
                      per_query={1: {"mrr": float("nan"), "hit@1": 1.0, "recall@100": 1.0}})
    mv.sanitize()
    _check("non-finite quality zeroed", all(math.isfinite(v) for v in mv.quality.values()))
    _check("any non-finite axis flags crashed", mv.crashed is True)
    _check("crashed candidate has zero primary (cannot win the gate)", mv.primary == 0.0)
    _check("non-finite latency repaired", math.isfinite(mv.latency_s))
    _check("per_query non-finite repaired",
           math.isfinite(mv.per_query[1]["mrr"]))


def test_sanitize_leaves_finite_vectors_untouched():
    mv = MetricVector(quality={"recall@20": 0.4, "hit@1": 0.2, "hit@5": 0.3, "mrr": 0.25},
                      latency_s=0.5, code_complexity=12.0)
    before = mv.to_dict()
    mv.sanitize()
    _check("finite vector unchanged by sanitize", mv.to_dict() == before)
    _check("finite vector not flagged crashed", mv.crashed is False)


def test_sanitize_blocks_inf_from_corrupting_dominance():
    """The concrete failure sanitize prevents: an inf on the primary axis would let
    a broken candidate DOMINATE every real program and take over the pool. After
    sanitize the same candidate can no longer dominate a genuine one."""
    good = MetricVector(quality={"recall@20": 0.4, "hit@1": 0.2, "hit@5": 0.3, "mrr": 0.25},
                        latency_s=0.5, code_complexity=12.0)
    raw = MetricVector(quality={"recall@20": float("inf"), "hit@1": 0, "hit@5": 0, "mrr": 0},
                       latency_s=0.0, code_complexity=0.0)
    _check("WITHOUT sanitize, an inf-primary candidate falsely dominates a real one",
           dominates(raw, good))
    raw.sanitize()
    _check("AFTER sanitize, it no longer dominates the real program",
           not dominates(raw, good))
    _check("AFTER sanitize, it cannot win the gate (primary is 0, flagged crashed)",
           raw.primary == 0.0 and raw.crashed is True)


# ---------------------------------------------------------- harness mechanics

# module-level (picklable) init/scorer for the process pool
def _init(arg):
    # `arg` is the flat config dict crossing the spawn boundary
    return {"factor": arg["factor"], "pid": os.getpid()}


def _score(component, task):
    return component["factor"] * task


def _init_counting(arg):
    # append our pid to a shared file every time init runs, to prove laziness
    with open(arg["marker"], "a") as f:
        f.write(f"{os.getpid()}\n")
    return {"factor": arg["factor"]}


def test_serial_path_when_one_worker():
    out = parallel_map([1, 2, 3], num_workers=1, init_fn=_init,
                       init_arg={"factor": 10}, scorer=_score)
    _check("serial path computes correctly", out == [10, 20, 30])


def test_parallel_crosses_config_and_preserves_order():
    tasks = list(range(8))
    out = parallel_map(tasks, num_workers=3, init_fn=_init,
                       init_arg={"factor": 7}, scorer=_score, max_tasks_per_child=4)
    _check("config (factor=7) crossed the spawn boundary into every worker",
           out == [7 * t for t in tasks])
    _check("results returned in input order", out == sorted(out))


def test_lazy_init_runs_once_per_worker():
    with tempfile.TemporaryDirectory() as d:
        marker = os.path.join(d, "inits.txt")
        # 1 worker, recycling disabled -> init must run exactly once for all 5 tasks
        out = parallel_map([1, 2, 3, 4, 5], num_workers=2, init_fn=_init_counting,
                           init_arg={"factor": 2, "marker": marker},
                           scorer=_score, max_tasks_per_child=None)
        _check("all tasks scored", out == [2, 4, 6, 8, 10])
        n_inits = len([l for l in open(marker) if l.strip()])
        # at most one init per spawned worker (<= num_workers), never per-task
        _check(f"lazy init ran once per worker, not per task (got {n_inits})",
               1 <= n_inits <= 2)


# -------------------------------------------------------------- budget guard

def test_budget_guard_forces_serial(capsys=None):
    from graphretr_opt.config import load_config
    from graphretr_opt.optimizer.fast_loop import FastLoop

    class _Budget:  # a stand-in active budget
        pass

    cfg = load_config(num_workers=4)
    loop_no_budget = FastLoop(cfg, None, None, None, None, None, None, budget=None)
    loop_budget = FastLoop(cfg, None, None, None, None, None, None, budget=_Budget())
    _check("parallel allowed when no OpenAI budget (minilm path)",
           loop_no_budget._parallel_scoring_enabled() == 4)
    _check("parallel refused (serial) when the OpenAI budget is active",
           loop_budget._parallel_scoring_enabled() == 1)
    cfg1 = load_config(num_workers=1)
    loop_serial = FastLoop(cfg1, None, None, None, None, None, None, budget=None)
    _check("num_workers=1 stays serial", loop_serial._parallel_scoring_enabled() == 1)


def main():
    test_sanitize_degrades_nonfinite_to_safe_worst()
    test_sanitize_leaves_finite_vectors_untouched()
    test_sanitized_crash_is_dominated_by_a_real_program()
    test_serial_path_when_one_worker()
    test_parallel_crosses_config_and_preserves_order()
    test_lazy_init_runs_once_per_worker()
    test_budget_guard_forces_serial()
    print("\nall parallel-foundation tests passed")


if __name__ == "__main__":
    main()
