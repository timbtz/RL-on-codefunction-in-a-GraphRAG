"""Unit tests for Phase 0.5: FastLoop._minibatch_ok (post-mortem #4).

Two fixes over the run-7 pre-screen, which false-killed 63% of candidates:
  1. Reference the monotone INCUMBENT `best` (the same bar the full gate uses),
     NOT a randomly-sampled pool parent.
  2. Promote on `value(child) >= value(best) - eps` (eps = minibatch_eps or
     1/mb_size), NOT strict `>`, so the b=20 SE ~ 0.11 stops false-killing ties
     and primary-axis-positive children.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_minibatch_prescreen
No FalkorDB / no network needed.
"""
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


def _mv(recall):
    return MetricVector(quality={"recall@20": recall, "hit@1": 0.0,
                                 "hit@5": 0.0, "mrr": 0.0})


class _Cache:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v


class _FakeReward:
    """score() keys the canned mv on program src (idxs ignored)."""

    def __init__(self, by_src):
        self._by = by_src

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        return self._by[src]


def _loop(reward, minibatch_eps=0.0):
    loop = FastLoop.__new__(FastLoop)
    loop._reward = reward
    # strict gate -> _gate_value(mv) == recall@20, so we control the scalar directly.
    loop._gate = Gate(mode="strict", metric="recall@20")
    loop._cfg = types.SimpleNamespace(gate_seed=42, probe_timeout_s=1.0,
                                      minibatch_eps=minibatch_eps)
    return loop


def _run_ok(loop, best, cand, reward, mb_size=5):
    gate_idxs = list(range(20))
    return loop._minibatch_ok(_Cache(), best, best, cand, cand,
                              gate_idxs, "fix", mb_size, loop._cfg)


def test_tie_passes_with_default_eps():
    best, cand = _prog("inc"), _prog("c")
    reward = _FakeReward({best.src: _mv(0.40), cand.src: _mv(0.40)})  # exact tie
    loop = _loop(reward)  # eps default = 1/5 = 0.2
    _check("exact tie passes the pre-screen (strict `>` would kill it)",
           _run_ok(loop, best, cand, reward) is True)


def test_within_eps_passes():
    best, cand = _prog("inc"), _prog("c")
    # child 0.04 below incumbent; default eps = 1/5 = 0.2 -> within band -> pass.
    reward = _FakeReward({best.src: _mv(0.40), cand.src: _mv(0.36)})
    loop = _loop(reward)
    _check("child within eps of the incumbent passes", _run_ok(loop, best, cand, reward))


def test_beyond_eps_fails():
    best, cand = _prog("inc"), _prog("c")
    # child 0.30 below incumbent; well beyond eps=0.2 -> genuine collapse -> fail.
    reward = _FakeReward({best.src: _mv(0.40), cand.src: _mv(0.10)})
    loop = _loop(reward)
    _check("a genuine collapse beyond eps is still cheap-killed",
           _run_ok(loop, best, cand, reward) is False)


def test_eps_config_override():
    best, cand = _prog("inc"), _prog("c")
    reward = _FakeReward({best.src: _mv(0.40), cand.src: _mv(0.25)})  # 0.15 below
    # default eps (1/5=0.2) would pass; tighten to 0.05 -> now fails.
    _check("minibatch_eps override tightens the band",
           _run_ok(_loop(reward, minibatch_eps=0.05), best, cand, reward) is False)
    # loosen to 0.30 -> passes.
    _check("minibatch_eps override widens the band",
           _run_ok(_loop(reward, minibatch_eps=0.30), best, cand, reward) is True)


def test_reference_is_incumbent_not_parent():
    """The screen must reference the INCUMBENT, not the sampled parent. We model
    a strong specialist parent and a weak incumbent: a child that clears the
    incumbent (but not the strong parent) must PASS -- the run-7 bug rejected it."""
    best, cand = _prog("weak_incumbent"), _prog("c")
    # incumbent 0.30; child 0.35 clears it. (A parent at 0.50 is irrelevant now:
    # _minibatch_ok no longer takes the parent.)
    reward = _FakeReward({best.src: _mv(0.30), cand.src: _mv(0.35)})
    loop = _loop(reward, minibatch_eps=0.01)
    _check("child clearing the incumbent passes regardless of any stronger parent",
           _run_ok(loop, best, cand, reward) is True)


def test_incumbent_score_cached_under_best_sha():
    best, cand = _prog("inc"), _prog("c")
    reward = _FakeReward({best.src: _mv(0.40), cand.src: _mv(0.40)})
    loop = _loop(reward)
    cache = _Cache()
    loop._minibatch_ok(cache, best, best, cand, cand, list(range(20)), "fix", 5, loop._cfg)
    _check("incumbent minibatch score cached under its own sha+epoch",
           f"mb:{best.sha}@fix" in cache.d)
    _check("child minibatch score cached under its own sha+epoch",
           f"mb:{cand.sha}@fix" in cache.d)


def main():
    test_tie_passes_with_default_eps()
    test_within_eps_passes()
    test_beyond_eps_fails()
    test_eps_config_override()
    test_reference_is_incumbent_not_parent()
    test_incumbent_score_cached_under_best_sha()
    print("\nall minibatch_prescreen tests passed")


if __name__ == "__main__":
    main()
