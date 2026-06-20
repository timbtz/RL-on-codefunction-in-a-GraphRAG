"""Unit tests for Phase 0.6b: cost-aware EXPORT re-pick in FastLoop._final_select
(post-mortem #5, Option-2 structure). Among finalists within `select_cost_floor`
of the top holdout quality value, ship the CHEAPEST by the deterministic
rerank_items meter (tiebreak code_complexity, then quality). The accept gate
stays pure-quality. floor=0 (default) => pure-quality argmax (run-7 parity).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_cost_repick
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


def _mv(recall=0.0, mrr=0.0, hit1=0.0, rerank=0.0, complexity=10.0):
    return MetricVector(
        quality={"recall@20": recall, "hit@1": hit1, "hit@5": 0.0, "mrr": mrr},
        rerank_items=rerank, code_complexity=complexity)


class _FakeReward:
    def __init__(self, by_src):
        self._by = by_src

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        return self._by[src]


class _FakeSubstrate:
    def __init__(self, holdout=(1, 2)):
        self._holdout = list(holdout)

    @property
    def meta_holdout_idxs(self):
        return list(self._holdout)

    def gate_idxs(self, run_dir, size, seed):
        return [0, 1, 2]


class _Pool:
    def __init__(self, programs):
        self.members = [types.SimpleNamespace(program=p) for p in programs]


def _loop(reward, select_cost_floor=0.0):
    loop = FastLoop.__new__(FastLoop)
    loop._reward = reward
    loop._budget = None
    loop._gate = Gate(mode="blend", blend={"recall@20": 0.4, "mrr": 0.3, "hit@1": 0.3})
    loop._cfg = types.SimpleNamespace(
        gate_size=2, gate_seed=42, probe_timeout_s=1.0,
        openai_budget_usd=5.0, select_holdout_n=0, select_cost_floor=select_cost_floor)
    return loop


def _ident(p):
    return p


def test_floor_zero_is_pure_quality_argmax():
    # B strictly best on quality but expensive; floor=0 -> quality wins anyway.
    a, b, c = _prog("a"), _prog("b"), _prog("c")
    reward = _FakeReward({
        a.src: _mv(recall=0.50, mrr=0.20, hit1=0.10, rerank=1.0),   # blend 0.29
        b.src: _mv(recall=0.45, mrr=0.40, hit1=0.30, rerank=99.0),  # blend 0.39 (costly)
        c.src: _mv(recall=0.40, mrr=0.10, hit1=0.05, rerank=1.0),   # blend 0.205
    })
    loop = _loop(reward, select_cost_floor=0.0)
    with tempfile.TemporaryDirectory() as d:
        export, _, _ = loop._final_select(_FakeSubstrate(), _Pool([a, b, c]),
                                          a, _mv(), d, _ident)
        _check("floor=0 keeps the pure-quality argmax (B) despite its high cost",
               export.sha == b.sha)
        audit = json.load(open(os.path.join(d, "select_holdout.json")))
        _check("audit: no cost re-pick when floor=0", audit["cost_repick"] is False)
        _check("audit records rerank_items per member",
               any("rerank_items" in m for m in audit["members"]))


def test_floor_repicks_cheaper_within_band():
    # B is quality-top (0.39) but very costly; A is within 0.10 (blend 0.29) and
    # 99x cheaper. floor=0.12 puts A in the band -> A ships.
    a, b, c = _prog("a"), _prog("b"), _prog("c")
    reward = _FakeReward({
        a.src: _mv(recall=0.50, mrr=0.20, hit1=0.10, rerank=1.0),   # blend 0.29
        b.src: _mv(recall=0.45, mrr=0.40, hit1=0.30, rerank=99.0),  # blend 0.39 top
        c.src: _mv(recall=0.40, mrr=0.10, hit1=0.05, rerank=1.0),   # blend 0.205 (out of band)
    })
    loop = _loop(reward, select_cost_floor=0.12)
    with tempfile.TemporaryDirectory() as d:
        export, mv, _ = loop._final_select(_FakeSubstrate(), _Pool([a, b, c]),
                                           a, _mv(), d, _ident)
        _check("cost-aware re-pick ships the cheap A within the quality band",
               export.sha == a.sha)
        _check("exported mv is the cheap member's", mv.rerank_items == 1.0)
        audit = json.load(open(os.path.join(d, "select_holdout.json")))
        _check("audit flags the cost re-pick", audit["cost_repick"] is True)
        _check("audit records the pure-quality top (B) separately",
               audit["quality_top_sha"] == b.sha and audit["exported_sha"] == a.sha)
        _check("audit records the floor", abs(audit["select_cost_floor"] - 0.12) < 1e-9)


def test_floor_does_not_reach_below_band():
    # C is cheapest of all but its quality (0.205) is outside the 0.05 band below
    # the top (0.39); it must NOT be picked.
    a, b, c = _prog("a"), _prog("b"), _prog("c")
    reward = _FakeReward({
        a.src: _mv(recall=0.50, mrr=0.20, hit1=0.10, rerank=50.0),  # blend 0.29 (out, >0.05 below)
        b.src: _mv(recall=0.45, mrr=0.40, hit1=0.30, rerank=50.0),  # blend 0.39 top
        c.src: _mv(recall=0.40, mrr=0.10, hit1=0.05, rerank=0.1),   # blend 0.205 cheapest but low Q
    })
    loop = _loop(reward, select_cost_floor=0.05)
    with tempfile.TemporaryDirectory() as d:
        export, _, _ = loop._final_select(_FakeSubstrate(), _Pool([a, b, c]),
                                          a, _mv(), d, _ident)
        _check("a cheap-but-low-quality member outside the band is not shipped",
               export.sha == b.sha)


def test_cost_tie_breaks_to_complexity():
    # Two members tie on quality AND cost -> the simpler (lower cc) ships.
    d_, e_ = _prog("d"), _prog("e")
    reward = _FakeReward({
        d_.src: _mv(recall=0.50, rerank=5.0, complexity=30.0),  # blend 0.20
        e_.src: _mv(recall=0.50, rerank=5.0, complexity=5.0),   # blend 0.20
    })
    loop = _loop(reward, select_cost_floor=0.0)
    with tempfile.TemporaryDirectory() as d:
        export, _, _ = loop._final_select(_FakeSubstrate(), _Pool([d_, e_]),
                                          d_, _mv(), d, _ident)
        _check("quality+cost tie breaks toward the simpler program", export.sha == e_.sha)


def main():
    test_floor_zero_is_pure_quality_argmax()
    test_floor_repicks_cheaper_within_band()
    test_floor_does_not_reach_below_band()
    test_cost_tie_breaks_to_complexity()
    print("\nall cost_repick tests passed")


if __name__ == "__main__":
    main()
