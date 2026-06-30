"""Phase 1: live-pool checkpoint/resume + graceful shutdown + atomic writes.

Covers the four mechanisms the production-improvements plan ports from openEvolve
(minus its three bugs):
  * CandidatePool / MetricVector serialization round-trip (incl. per_query int keys)
  * atomic_write_json (temp + os.replace; no torn file on a mid-write crash)
  * a real FastLoop run -> checkpoint.json -> a second FastLoop that RESUMES at the
    next step, appends to lineage.jsonl, and restores pool + counters
  * the resume guard refuses a checkpoint written under a different experiment

No FalkorDB / network / MLflow needed -- reuses the fake loop harness shape.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_checkpoint_resume.py
"""
import contextlib
import json
import os
import tempfile
from dataclasses import replace

from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.atomic_io import atomic_write_json
from graphretr_opt.config import load_config
from graphretr_opt.optimizer.edit_budget import EditBudget
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.optimizer.pool import CandidatePool
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


# --------------------------------------------------------------- unit: round-trips

def test_metricvector_round_trip_coerces_per_query_keys():
    mv = MetricVector(quality={"recall@20": 0.4, "hit@1": 0.1, "hit@5": 0.2, "mrr": 0.3},
                      latency_s=1.5, code_complexity=12.0,
                      per_query={3: {"mrr": 0.9, "hit@1": 1.0, "recall@100": 1.0},
                                 7: {"mrr": 0.0, "hit@1": 0.0, "recall@100": 0.0}})
    # simulate the JSON round-trip (int keys become strings)
    restored = MetricVector.from_dict(json.loads(json.dumps(mv.to_dict())))
    _check("quality preserved", restored.quality == mv.quality)
    _check("scalar axes preserved",
           restored.latency_s == 1.5 and restored.code_complexity == 12.0)
    _check("per_query keys coerced back to int (not str)",
           set(restored.per_query) == {3, 7})
    _check("per_query payload preserved",
           restored.per_query[3]["mrr"] == 0.9)


def test_pool_round_trip():
    pool = CandidatePool(cap=5)
    for i, r in enumerate((0.30, 0.42, 0.35)):
        prog = SearchProgram(f"def search(q, G):\n    return [{i}]\n", family="fam")
        mv = MetricVector(quality={"recall@20": r, "hit@1": 0.1, "hit@5": 0.1, "mrr": r / 2},
                          per_query={i: {"mrr": r / 2, "hit@1": 0.0, "recall@100": 1.0}})
        pool.consider(prog, mv)
    before_shas = pool.shas()
    blob = json.loads(json.dumps(pool.to_dict()))  # force the JSON boundary
    restored = CandidatePool.from_dict(blob)
    _check("cap preserved", restored.cap == 5)
    _check("members preserved by sha", restored.shas() == before_shas)
    _check("metrics + primary preserved",
           round(restored.members[0].metrics.primary, 4)
           == round(pool.members[0].metrics.primary, 4))


def test_pool_from_dict_drops_uncompilable_members():
    pool = CandidatePool(cap=5)
    good = SearchProgram("def search(q, G):\n    return [1]\n", family="fam")
    pool.consider(good, MetricVector(quality={"recall@20": 0.4, "hit@1": 0, "hit@5": 0, "mrr": 0.2}))
    blob = pool.to_dict()
    blob["members"].append({"src": "this is not python", "family": "fam",
                            "metrics": MetricVector().to_dict(), "children": 0})
    restored = CandidatePool.from_dict(blob, validate=lambda src: "def search" in src)
    _check("uncompilable member dropped on resume", len(restored) == 1)
    _check("good member kept", good.sha in restored.shas())


# --------------------------------------------------------------- unit: atomic write

def test_atomic_write_is_all_or_nothing():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.json")
        atomic_write_json(path, {"a": 1})
        _check("file written", json.load(open(path)) == {"a": 1})
        atomic_write_json(path, {"a": 2, "b": 3})
        _check("overwrite is complete", json.load(open(path)) == {"a": 2, "b": 3})
        _check("no leftover tmp files",
               [f for f in os.listdir(d) if ".tmp" in f] == [])


# --------------------------------------------------------------- integration harness

SEED_SRC = "def search(q, G):\n    return []\n"


def _row():
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
    def __init__(self, start=0.30):
        self._r = start

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        if return_rows:
            return None, [_row()]
        self._r = round(self._r + 0.01, 4)
        return MetricVector(quality={"recall@20": self._r, "hit@1": 0.1,
                                     "hit@5": 0.1, "mrr": 0.1},
                            code_complexity=10.0,
                            per_query={0: {"mrr": 0.1, "hit@1": 0.0, "recall@100": 1.0}})


class _FakeMutator:
    def __init__(self):
        self._n = 0
        self.propose_calls = 0  # discriminates resume (skips done steps) from fresh

    def propose(self, prog, fails, wins, recent, L_t, plateau=False,
                validate=None, repair_budget=0, accepted_entries=None,
                combine_with=None, summary=None):
        self._n += 1
        self.propose_calls += 1
        cand = prog.with_src(prog.src + f"\n# edit {self._n}")
        if validate is not None:
            validate(cand)
        return cand, f"transcript {self._n}", {"reject_reason": None, "probe_failed": 0}

    @property
    def call_counts(self):
        return {}


class _FakeTracker:
    def log_vector(self, *a, **k):
        pass

    def log_metrics(self, *a, **k):
        pass

    @contextlib.contextmanager
    def step_span(self, name, inputs=None):
        yield None

    def record_step(self, span, outputs=None, attributes=None):
        pass


class _FakeSubstrate:
    train_idxs = list(range(10))
    meta_holdout_idxs = []

    def gate_idxs(self, run_dir, size, seed):
        return [0, 1]

    def example(self, i):
        return (f"query-{i}", None)


def _cfg(root, **kw):
    base = dict(root=root, rollout_batch=2, reflect_top=2, gate_rotate_every=0,
                gate_mode="strict", gate_metric="recall@20", stop_after_stale=0,
                architect_plateau=0, edit_schedule="const", max_edits=2, min_edits=1,
                gate_size=2, gate_seed=42, probe_timeout_s=1.0)
    base.update(kw)
    return load_config(**base)


def _make_loop(cfg, steps, reward=None, mutator=None):
    return FastLoop(cfg, _FakeGraph(), _FakeSandbox(), reward or _FakeReward(),
                    mutator or _FakeMutator(), EditBudget("const", 2, 1, steps),
                    _FakeTracker(), budget=None)


def test_checkpoint_then_resume():
    with tempfile.TemporaryDirectory() as d:
        # Run 1: 2 steps, checkpoint every step.
        cfg1 = _cfg(d, steps=2, checkpoint_every=1, pool_enabled=True, pool_cap=8)
        seed = SearchProgram(SEED_SRC, family="test")
        _make_loop(cfg1, 2).run(_FakeSubstrate(), seed, 2, "ckpt")

        cpath = os.path.join(d, "runs", "ckpt", "checkpoint.json")
        _check("checkpoint.json written", os.path.exists(cpath))
        blob = json.load(open(cpath))
        _check("checkpoint last_step is the final completed step (1)",
               blob["last_step"] == 1)
        _check("checkpoint carries pool + best + counters",
               "pool" in blob and blob["best"]["src"]
               and "n_accepted" in blob)

        lpath = os.path.join(d, "runs", "ckpt", "lineage.jsonl")
        rows_after_run1 = [json.loads(l) for l in open(lpath) if l.strip()]
        _check("run 1 wrote 2 lineage rows", len(rows_after_run1) == 2)

        # Run 2: same campaign, resume=True, total budget 4 steps -> should run 2,3.
        cfg2 = _cfg(d, steps=4, checkpoint_every=1, resume=True,
                    pool_enabled=True, pool_cap=8)
        mut2 = _FakeMutator()
        # A fresh reward continues climbing; resume must NOT re-run steps 0,1.
        _make_loop(cfg2, 4, reward=_FakeReward(start=0.40), mutator=mut2).run(
            _FakeSubstrate(), seed, 4, "ckpt")

        rows_after_run2 = [json.loads(l) for l in open(lpath) if l.strip()]
        steps_seen = [r["step"] for r in rows_after_run2]
        # The decisive resume check: a fresh run would propose 4 times and truncate
        # lineage to [0,1,2,3] too -- identical rows. What ONLY resume can do is
        # skip the already-done steps, so the resumed mutator proposes exactly twice.
        _check("resume skipped completed steps (proposed only for 2,3)",
               mut2.propose_calls == 2)
        _check("lineage appended across runs (4 rows total)",
               len(rows_after_run2) == 4)
        _check("steps are 0,1,2,3 contiguous (run1's 0,1 + run2's 2,3)",
               steps_seen == [0, 1, 2, 3])
        blob2 = json.load(open(cpath))
        _check("checkpoint advanced to last_step=3", blob2["last_step"] == 3)


def test_resume_guard_rejects_foreign_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        cfg1 = _cfg(d, steps=2, checkpoint_every=1, gate_seed=42)
        seed = SearchProgram(SEED_SRC, family="test")
        _make_loop(cfg1, 2).run(_FakeSubstrate(), seed, 2, "guard")
        # A different experiment (gate_seed changed) must refuse to resume.
        cfg2 = _cfg(d, steps=4, resume=True, gate_seed=999)
        loop2 = _make_loop(cfg2, 4)
        _check("foreign checkpoint is rejected by the guard",
               loop2._load_checkpoint(os.path.join(d, "runs", "guard")) is None)
        # but the matching-experiment checkpoint loads
        cfg3 = _cfg(d, steps=4, resume=True, gate_seed=42)
        loop3 = _make_loop(cfg3, 4)
        _check("matching checkpoint loads",
               loop3._load_checkpoint(os.path.join(d, "runs", "guard")) is not None)


def main():
    test_metricvector_round_trip_coerces_per_query_keys()
    test_pool_round_trip()
    test_pool_from_dict_drops_uncompilable_members()
    test_atomic_write_is_all_or_nothing()
    test_checkpoint_then_resume()
    test_resume_guard_rejects_foreign_checkpoint()
    print("\nall checkpoint/resume tests passed")


if __name__ == "__main__":
    main()
