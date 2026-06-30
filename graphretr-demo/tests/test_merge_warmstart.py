"""Unit tests for the merge warm-start primitive (archipelago tournament-merge).

Covers:
  (a) the pool admits >=2 distinct champions and select_mate returns the partner
      (so COMBINE has a second parent to synthesize with);
  (b) a champion that exceeds the token wall is rejected by it but admitted when
      the wall is lifted (the warm-start mitigation);
  (c) FastLoop.run(seed_pool=[A, B]) end-to-end (steps=0) warm-starts both
      champions into the pool and exports the stronger one -- with merge=True
      forcing COMBINE mode from step 0.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_merge_warmstart
No FalkorDB / no network needed.
"""
import contextlib
import io
import os
import random
import tempfile
import types

from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.env.null_sandbox import NullGraph, NullSandbox
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.optimizer.gate import Gate
from graphretr_opt.optimizer.pool import CandidatePool
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _prog(tag):
    return SearchProgram(f"def search(q, G):\n    return {{{tag!r}: 1.0}}\n", family="t")


def _mv(recall, mrr, per_query, tokens=10.0):
    return MetricVector(quality={"recall@20": recall, "mrr": mrr},
                        code_complexity=10.0, code_tokens=tokens,
                        per_query=per_query)


# A and B are mutually non-dominated on the breeding signal: A has higher
# recall@20 (the dominance primary) but B is sole-best on query 1 (per_query
# mrr), so B survives as a specialist. Seed is sole-best on nothing -> evicted.
_SEED_MV = _mv(0.30, 0.30, {0: {"mrr": 0.0}, 1: {"mrr": 0.0}, 2: {"mrr": 0.0}})
_A_MV = _mv(0.50, 0.40, {0: {"mrr": 0.9}, 1: {"mrr": 0.1}, 2: {"mrr": 0.1}})
_B_MV = _mv(0.45, 0.45, {0: {"mrr": 0.1}, 1: {"mrr": 0.9}, 2: {"mrr": 0.1}})


def test_pool_admits_two_and_mate_returns_partner():
    pool = CandidatePool(cap=24, max_tokens=0.0)
    seed, a, b = _prog("seed"), _prog("A"), _prog("B")
    pool.consider(seed, _SEED_MV)
    pool.consider(a, _A_MV)
    pool.consider(b, _B_MV)
    _check("exactly two distinct champions in the pool", len(pool) == 2)
    _check("A admitted", a.sha in pool.shas())
    _check("B admitted as a specialist", b.sha in pool.shas())
    mate = pool.select_mate(random.Random(0), exclude_sha=a.sha)
    _check("select_mate(exclude=A) returns the partner B",
           mate is not None and mate.sha == b.sha)


def test_token_wall_lift_admits_oversized_champion():
    big = _prog("big")
    big_mv = _mv(0.60, 0.50, {0: {"mrr": 0.9}}, tokens=10_000.0)
    walled = CandidatePool(cap=24, max_tokens=100.0)
    _, admitted = walled.consider(big, big_mv)
    _check("oversized champion rejected by the token wall", admitted is False)
    lifted = CandidatePool(cap=24, max_tokens=100.0)
    saved = lifted.max_tokens
    lifted.max_tokens = 0.0                       # the warm-start lift
    _, admitted2 = lifted.consider(big, big_mv)
    lifted.max_tokens = saved                     # restore for later mutations
    _check("oversized champion admitted when the wall is lifted", admitted2 is True)
    _check("token wall restored after the lift", lifted.max_tokens == 100.0)


class _Reward:
    """score() looks the program's src up in a {src -> MetricVector} table."""

    def __init__(self, by_src):
        self._by = by_src

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        return self._by[src]


class _Sub:
    train_idxs = list(range(5))
    meta_holdout_idxs = []                         # empty -> _final_select gate fallback
    promote_idxs = []

    def gate_idxs(self, run_dir, size, seed):
        return [0, 1, 2]

    def example(self, i):
        return (f"q{i}", None)                     # _probe_queries reads [0]


class _Tracker:
    def __init__(self):
        self.metrics = {}

    def log_vector(self, *a, **k):
        pass

    def log_metrics(self, d, step=None):
        self.metrics.update(d)


def _loop(reward, cfg, tracker):
    loop = FastLoop.__new__(FastLoop)             # bypass __init__ (no infra)
    loop._cfg = cfg
    loop._graph = NullGraph()
    loop._sandbox = NullSandbox()
    loop._reward = reward
    loop._mutator = types.SimpleNamespace(call_counts={})
    loop._tracker = tracker
    loop._budget = None
    loop._rows_by_sha = {}
    loop._gate = Gate(mode="strict", metric="recall@20")
    return loop


def test_run_with_seed_pool_warmstarts_both_and_forces_combine():
    seed, a, b = _prog("seed"), _prog("A"), _prog("B")
    reward = _Reward({seed.src: _SEED_MV, a.src: _A_MV, b.src: _B_MV})
    with tempfile.TemporaryDirectory() as d:
        cfg = types.SimpleNamespace(
            runs_dir=d, gate_rotate_every=0, gate_size=3, gate_seed=42,
            probe_timeout_s=1.0, pool_enabled=True, pool_cap=24,
            gate_max_tokens=0.0, minibatch_size=0, meta_eval_every=0,
            promote_margin=0.0, max_generations=1, merge=True,
            architect_plateau=0, stop_after_stale=0, checkpoint_every=0,
            resume=False, openai_budget_usd=0.0, select_holdout_n=0,
            select_cost_floor=0.0, select_timeout_s=60.0, gate_mode="strict",
            gate_metric="recall@20", buffer_last=8, num_workers=1,
            rollout_fanout=1, pool_discount=True)
        tracker = _Tracker()
        loop = _loop(reward, cfg, tracker)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = loop.run(_Sub(), seed, 0, "mergecamp", seed_pool=[a, b])
        out = buf.getvalue()

        _check("both warm-start champions admitted (pool_size_final == 2)",
               tracker.metrics.get("pool_size_final") == 2)
        _check("warm-start log shows A admitted",
               f"merge warm-start {a.sha[:8]}" in out and "admitted=True" in out)
        _check("warm-start log shows B admitted",
               f"merge warm-start {b.sha[:8]}" in out)
        _check("'combine inert' warning did NOT fire (>=2 admitted)",
               "combine inert" not in out)
        _check("the stronger champion A is exported",
               result.best_program.sha == a.sha)
        _check("best_search.py written", os.path.exists(
            os.path.join(d, "mergecamp", "best_search.py")))


def _run():
    test_pool_admits_two_and_mate_returns_partner()
    test_token_wall_lift_admits_oversized_champion()
    test_run_with_seed_pool_warmstarts_both_and_forces_combine()
    print("all merge-warmstart tests passed")


if __name__ == "__main__":
    _run()
