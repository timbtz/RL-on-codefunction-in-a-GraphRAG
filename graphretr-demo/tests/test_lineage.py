"""Integration test for Phase C: the consolidated lineage.jsonl.

Drives the REAL FastLoop.run over a fixture loop (fake graph/sandbox/reward/
mutator -- no DB, no network) and asserts the lineage record it writes: exactly
one row per step, accepted and rejected alike, with a parent->child sha chain.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_lineage
"""
import contextlib
import json
import os
import tempfile

from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.config import load_config
from graphretr_opt.optimizer.edit_budget import EditBudget
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


SEED_SRC = "def search(q, G):\n    return []\n"


def _row():
    """One rollout row shaped for FastLoop._reflect (a 'missed' failure)."""
    return {"query": "q", "answer_ids": [1], "retrieved": [(2, 0.5)],
            "gold_ranks": {}, "error": None,
            "metrics": {"recall@20": 0.5, "hit@1": 0.0, "hit@5": 0.0, "mrr": 0.2}}


class _FakeGraph:
    def get_text(self, ids, limit=50):
        return {i: f"text-{i}" for i in ids}


class _FakeSandbox:
    def compile(self, src):
        return ("fn", src)

    def probe(self, fn, probes, timeout):
        return None


class _FakeReward:
    """Each gate score strictly beats the last -> every candidate is accepted,
    so the lineage chain grows one accepted link per step."""

    def __init__(self):
        self._r = 0.30

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        if return_rows:
            return None, [_row()]
        self._r = round(self._r + 0.01, 4)
        return MetricVector(quality={"recall@20": self._r, "hit@1": 0.1,
                                     "hit@5": 0.1, "mrr": 0.1},
                            code_complexity=10.0)


class _FakeMutator:
    def __init__(self):
        self._n = 0

    def propose(self, prog, fails, wins, recent, L_t, plateau=False,
                validate=None, repair_budget=0, accepted_entries=None):
        self._n += 1
        cand = prog.with_src(prog.src + f"\n# edit {self._n}")
        if validate is not None:
            validate(cand)  # mirror real propose: compile+probe, populate fn_cache
        return cand, f"transcript {self._n}", {"reject_reason": None,
                                               "probe_failed": 0}

    @property
    def call_counts(self):
        return {}


class _FakeTracker:
    def log_vector(self, *a, **k):
        pass

    def log_metrics(self, *a, **k):
        pass

    # Phase G tracing seams: the loop calls these unconditionally; mirror the real
    # tracker's disabled contract (step_span yields None, record_step is a no-op).
    @contextlib.contextmanager
    def step_span(self, name, inputs=None):
        yield None

    def record_step(self, span, outputs=None, attributes=None):
        pass


class _FakeSubstrate:
    train_idxs = list(range(10))

    def gate_idxs(self, run_dir, size, seed):
        return [0, 1]

    def example(self, i):
        return (f"query-{i}", None)


def test_lineage_chain():
    steps = 3
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(root=d, rollout_batch=2, reflect_top=2,
                          gate_rotate_every=0, gate_mode="strict",
                          gate_metric="recall@20", stop_after_stale=0,
                          architect_plateau=0, edit_schedule="const",
                          max_edits=2, min_edits=1, gate_size=2, gate_seed=42,
                          probe_timeout_s=1.0)
        seed = SearchProgram(SEED_SRC, family="test")
        loop = FastLoop(cfg, _FakeGraph(), _FakeSandbox(), _FakeReward(),
                        _FakeMutator(), EditBudget("const", 2, 1, steps),
                        _FakeTracker(), budget=None)
        loop.run(_FakeSubstrate(), seed, steps, "lineage_test")

        path = os.path.join(d, "runs", "lineage_test", "lineage.jsonl")
        _check("lineage.jsonl written", os.path.exists(path))
        rows = [json.loads(l) for l in open(path) if l.strip()]

        _check("exactly one row per step", len(rows) == steps)
        _check("steps are 0..N-1 in order",
               [r["step"] for r in rows] == list(range(steps)))
        _check("every step accepted in this fixture",
               all(r["accepted"] for r in rows))

        _check("row 0 parent is the seed sha", rows[0]["parent_sha"] == seed.sha)
        chain_ok = all(rows[k]["parent_sha"] == rows[k - 1]["child_sha"]
                       for k in range(1, len(rows)))
        _check("parent->child sha chain is contiguous", chain_ok)
        _check("each accepted edit actually changed the sha",
               all(r["child_sha"] and r["child_sha"] != r["parent_sha"] for r in rows))

        r0 = rows[0]
        _check("metric_vector recorded as flat dict",
               isinstance(r0["metric_vector"], dict)
               and "quality_recall_at_20" in r0["metric_vector"])
        _check("change_summary is non-empty", bool(r0["change_summary"]))
        _check("edit_budget recorded", r0["edit_budget"] == 2)
        _check("gate_tag recorded", r0["gate_tag"] == "fix")
        # run-6: acceptance is pool admission; the reason names why it was kept.
        _check("reason present for accepted row",
               "admitted to pool" in r0["reason"])
        _check("pool/admission bookkeeping present in lineage",
               r0.get("admitted") is True and "pool_size" in r0)


def test_lineage_records_rejects():
    """A mutator that returns no candidate still yields one row per step
    (rejected, child_sha=None) -- dead-ends are kept, not dropped."""
    steps = 2

    class _NoCandMutator:
        def propose(self, *a, **k):
            return None, "no candidate", {"reject_reason": "parse",
                                          "probe_failed": 0}

        @property
        def call_counts(self):
            return {}

    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(root=d, rollout_batch=2, reflect_top=2,
                          gate_rotate_every=0, stop_after_stale=0,
                          architect_plateau=0, gate_seed=42, probe_timeout_s=1.0)
        seed = SearchProgram(SEED_SRC, family="test")
        loop = FastLoop(cfg, _FakeGraph(), _FakeSandbox(), _FakeReward(),
                        _NoCandMutator(), EditBudget("const", 2, 1, steps),
                        _FakeTracker(), budget=None)
        loop.run(_FakeSubstrate(), seed, steps, "lineage_reject")

        path = os.path.join(d, "runs", "lineage_reject", "lineage.jsonl")
        rows = [json.loads(l) for l in open(path) if l.strip()]
        _check("rejected steps still produce one row each", len(rows) == steps)
        _check("rejected rows carry no child sha",
               all(r["child_sha"] is None and not r["accepted"] for r in rows))
        _check("rejected parent stays the incumbent seed",
               all(r["parent_sha"] == seed.sha for r in rows))


def main():
    test_lineage_chain()
    test_lineage_records_rejects()
    print("\nall lineage tests passed")


if __name__ == "__main__":
    main()
