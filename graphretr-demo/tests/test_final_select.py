"""Unit tests for Phase 1: FastLoop._final_select -- the final held-out bake-off
that fixes the run-7 export bug (exporting the rotating-gate "last belt-holder"
instead of re-scoring every survivor on ONE fixed set and exporting the argmax).

Covers:
  (a) the per-epoch "headline" winner != the fixed-holdout winner -> the holdout
      winner is exported; and a within-noise tie is broken by code_complexity
      (the simpler program wins -- read via getattr, NOT mv.get()).
  (b) empty meta-holdout -> the fixed-gate fallback path is used.
  (c) a member that raises on the holdout is excluded, not fatal.
  (d) budget exhausted mid-bake-off -> fall back to the best scored so far.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_final_select
No FalkorDB / no network needed.
"""
import json
import os
import tempfile
import types

from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.optimizer.gate import Gate
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _prog(tag):
    return SearchProgram(f"def search(q, G):\n    return {{{tag!r}: 1.0}}\n", family="t")


def _mv(recall=0.0, mrr=0.0, hit1=0.0, complexity=10.0, crashed=False):
    mv = MetricVector(
        quality={"recall@20": recall, "hit@1": hit1, "hit@5": 0.0, "mrr": mrr},
        code_complexity=complexity)
    mv.crashed = crashed
    return mv


class _FakeReward:
    """score() looks the program's src up in a {src -> mv | Exception} table; a
    mapped Exception is raised (the crash-on-holdout case)."""

    def __init__(self, by_src):
        self._by = by_src

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        v = self._by[src]
        if isinstance(v, Exception):
            raise v
        return v


class _FakeSubstrate:
    def __init__(self, holdout, gate_idxs=(0, 1, 2)):
        self._holdout = list(holdout)
        self._gate = list(gate_idxs)
        self.gate_idxs_called = False

    @property
    def meta_holdout_idxs(self):
        return list(self._holdout)

    def gate_idxs(self, run_dir, size, seed):
        self.gate_idxs_called = True
        return list(self._gate)


class _Pool:
    def __init__(self, programs):
        self.members = [types.SimpleNamespace(program=p) for p in programs]


class _FakeBudget:
    """spent_usd pops a value per access so we can cross the ceiling mid-bake-off
    (the proactive guard reads spent_usd once before each member)."""

    def __init__(self, vals):
        self._vals = list(vals)

    @property
    def spent_usd(self):
        return self._vals.pop(0) if len(self._vals) > 1 else self._vals[0]


def _loop(reward, cfg=None, budget=None):
    loop = FastLoop.__new__(FastLoop)          # bypass __init__ (no infra needed)
    loop._reward = reward
    loop._budget = budget
    # the run-7 blend gate: recall@20:0.4, mrr:0.3, hit@1:0.3
    loop._gate = Gate(mode="blend", blend={"recall@20": 0.4, "mrr": 0.3, "hit@1": 0.3})
    loop._cfg = cfg or types.SimpleNamespace(
        gate_size=2, gate_seed=42, probe_timeout_s=1.0,
        openai_budget_usd=5.0, select_holdout_n=0)
    return loop


def _ident(p):
    return p  # get_fn: the fake reward ignores the fn, keys on src


def test_holdout_winner_beats_gate_headline():
    a, b, c = _prog("a"), _prog("b"), _prog("c")
    # A is the rotating-gate "headline" incumbent we pass in, but on the FIXED
    # holdout B has the highest blend value -> B must be exported.
    reward = _FakeReward({
        a.src: _mv(recall=0.50, mrr=0.20, hit1=0.10),   # blend 0.29
        b.src: _mv(recall=0.45, mrr=0.40, hit1=0.30),   # blend 0.39 <- best
        c.src: _mv(recall=0.40, mrr=0.10, hit1=0.05),   # blend 0.205
    })
    loop = _loop(reward)
    sub = _FakeSubstrate(holdout=[10, 11, 12])
    pool = _Pool([a, b, c])
    with tempfile.TemporaryDirectory() as d:
        export, mv, scored = loop._final_select(sub, pool, a, reward.score(None, [], src=a.src),
                                                 d, _ident)
        _check("holdout winner B exported, not gate-headline A", export.sha == b.sha)
        _check("returned mv is B's holdout mv", abs(mv.get("mrr") - 0.40) < 1e-9)
        _check("all three members scored", len(scored) == 3)
        _check("scored sorted by blend value desc",
               scored[0][1].sha == b.sha and scored[-1][1].sha == c.sha)
        audit = json.load(open(os.path.join(d, "select_holdout.json")))
        _check("audit exported_sha == B", audit["exported_sha"] == b.sha)
        _check("audit records gate headline A", audit["gate_headline_sha"] == a.sha)
        _check("audit flags the export changed", audit["export_changed"] is True)
        _check("audit lists every member", len(audit["members"]) == 3)


def test_tie_broken_by_code_complexity():
    d_, e_ = _prog("d"), _prog("e")
    # identical blend value; E is the simpler program (lower code_complexity).
    reward = _FakeReward({
        d_.src: _mv(recall=0.50, mrr=0.0, hit1=0.0, complexity=30.0),  # blend 0.20
        e_.src: _mv(recall=0.50, mrr=0.0, hit1=0.0, complexity=5.0),   # blend 0.20
    })
    loop = _loop(reward)
    with tempfile.TemporaryDirectory() as d:
        export, _, scored = loop._final_select(
            _FakeSubstrate(holdout=[1, 2]), _Pool([d_, e_]), d_,
            reward.score(None, [], src=d_.src), d, _ident)
        _check("tie broken toward the simpler program (lower code_complexity)",
               export.sha == e_.sha)
        _check("both tied members still scored", len(scored) == 2)


def test_empty_holdout_uses_gate_fallback():
    a = _prog("a")
    reward = _FakeReward({a.src: _mv(recall=0.4, mrr=0.2, hit1=0.1)})
    loop = _loop(reward)
    sub = _FakeSubstrate(holdout=[])            # holdout disabled
    with tempfile.TemporaryDirectory() as d:
        export, _, scored = loop._final_select(sub, _Pool([a]), a,
                                               reward.score(None, [], src=a.src), d, _ident)
        _check("empty holdout falls back to the fixed gate", sub.gate_idxs_called)
        _check("fallback still selects a winner", export.sha == a.sha and len(scored) == 1)


def test_member_that_raises_is_excluded_not_fatal():
    a, b, c = _prog("a"), _prog("b"), _prog("c")
    reward = _FakeReward({
        a.src: _mv(recall=0.40, mrr=0.20, hit1=0.10),       # blend 0.25
        b.src: RuntimeError("boom on holdout"),             # raises -> excluded
        c.src: _mv(recall=0.50, mrr=0.30, hit1=0.20),       # blend 0.35 <- best survivor
    })
    loop = _loop(reward)
    with tempfile.TemporaryDirectory() as d:
        export, _, scored = loop._final_select(
            _FakeSubstrate(holdout=[1, 2]), _Pool([a, b, c]), a,
            reward.score(None, [], src=a.src), d, _ident)
        _check("crashing member excluded (2 of 3 scored)", len(scored) == 2)
        _check("best surviving member exported", export.sha == c.sha)
        _check("crashing member absent from scored",
               b.sha not in {p.sha for _, p, _ in scored})


def test_budget_exhausted_falls_back_to_best_so_far():
    a, b, c = _prog("a"), _prog("b"), _prog("c")
    reward = _FakeReward({
        a.src: _mv(recall=0.40, mrr=0.20, hit1=0.10),
        b.src: _mv(recall=0.99, mrr=0.99, hit1=0.99),  # would win, but never scored
        c.src: _mv(recall=0.99, mrr=0.99, hit1=0.99),
    })
    # ceiling 5.0; spent jumps over it after the first member is scored.
    budget = _FakeBudget([0.0, 999.0])
    loop = _loop(reward, budget=budget)
    seed_best = _mv(recall=0.40, mrr=0.20, hit1=0.10)
    with tempfile.TemporaryDirectory() as d:
        export, _, scored = loop._final_select(
            _FakeSubstrate(holdout=[1, 2]), _Pool([a, b, c]), a, seed_best, d, _ident)
        _check("budget guard stops after the first member", len(scored) == 1)
        _check("falls back to the best scored BEFORE the ceiling (A)",
               export.sha == a.sha)


def test_no_members_keeps_incumbent():
    a = _prog("a")
    # every candidate raises -> nothing scored -> keep the incumbent unchanged.
    reward = _FakeReward({a.src: RuntimeError("boom")})
    loop = _loop(reward)
    seed_best = _mv(recall=0.4)
    with tempfile.TemporaryDirectory() as d:
        export, mv, scored = loop._final_select(
            _FakeSubstrate(holdout=[1, 2]), _Pool([a]), a, seed_best, d, _ident)
        _check("no scored members -> incumbent kept", export.sha == a.sha and scored == [])
        _check("no audit written when nothing scored",
               not os.path.exists(os.path.join(d, "select_holdout.json")))


def main():
    test_holdout_winner_beats_gate_headline()
    test_tie_broken_by_code_complexity()
    test_empty_holdout_uses_gate_fallback()
    test_member_that_raises_is_excluded_not_fatal()
    test_budget_exhausted_falls_back_to_best_so_far()
    test_no_members_keeps_incumbent()
    print("\nall final_select tests passed")


if __name__ == "__main__":
    main()
