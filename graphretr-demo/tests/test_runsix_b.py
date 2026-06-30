"""Unit tests for Phase B: the minibatch pre-screen decision, the meta-holdout
partition (disjoint from the gate pool, never sampled by the gate), and the
generation-vs-ranking failure attribution fed to the mutator (B1/C2).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_runsix_b
No FalkorDB / no network needed.
"""
from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.config import load_config
from starksearch.qa import Substrate
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.optimizer.gate import Gate
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


# ----------------------------------------------------------- minibatch (B2)

class _Cache:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v


class _MBReward:
    """Returns a metric vector whose recall@20/mrr is encoded in the program src
    marker, so the minibatch comparison is deterministic and DB-free."""

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        s = src or ""
        # PARENT=0.4; BETTER above it; WORSE clearly below the eps band (1/20=0.05
        # at b=20) so it is screened. A mere tie is promoted by design now
        # (post-mortem #4: the screen is `>= best - eps`, not strict `>`), so WORSE
        # must be genuinely lower, not equal to the parent.
        r = 0.6 if "CAND_BETTER" in s else (0.2 if "CAND_WORSE" in s else 0.4)
        return MetricVector(quality={"recall@20": r, "hit@1": 0.0,
                                     "hit@5": 0.0, "mrr": r})


def _mb_loop():
    loop = FastLoop.__new__(FastLoop)
    loop._reward = _MBReward()
    loop._gate = Gate(mode="blend", blend={"recall@20": 0.4, "mrr": 0.3, "hit@1": 0.3})
    return loop


def test_minibatch_promotion():
    cfg = load_config()
    loop = _mb_loop()
    gate_idxs = list(range(50))
    parent = SearchProgram("def search(q, G):\n    return {1: 1.0}  # PARENT\n")
    better = SearchProgram("def search(q, G):\n    return {1: 1.0}  # CAND_BETTER\n")
    worse = SearchProgram("def search(q, G):\n    return {1: 1.0}  # CAND_WORSE\n")

    ok = loop._minibatch_ok(_Cache(), parent, "pfn", better, "cfn",
                            gate_idxs, "fix", 20, cfg)
    _check("minibatch: child beating parent is promoted to the full gate", ok)
    no = loop._minibatch_ok(_Cache(), parent, "pfn", worse, "cfn",
                            gate_idxs, "fix", 20, cfg)
    _check("minibatch: child not beating parent is screened out", not no)

    # the parent minibatch score is cached per (sha, epoch): one score, reused
    cache = _Cache()
    loop._minibatch_ok(cache, parent, "pfn", better, "cfn", gate_idxs, "fix", 20, cfg)
    _check("minibatch: parent score cached per (sha, epoch)",
           f"mb:{parent.sha}@fix" in cache.d)


# ----------------------------------------------------------- meta-holdout (B3)

def test_meta_holdout_partition():
    s = Substrate.__new__(Substrate)        # bypass STaRK load
    s._val = list(range(2241))
    # promote off (pr_size=0): backward-compat -- holdout identical to the old
    # 2-slice partition, the rest is the gate pool.
    holdout, promote, pool = s._partition(300, 1234, 0, 5678)
    _check("meta: holdout has the requested size", len(holdout) == 300)
    _check("meta: promote empty when pr_size=0", promote == [])
    _check("meta: holdout and gate pool are DISJOINT",
           set(holdout).isdisjoint(set(pool)))
    _check("meta: holdout + gate pool == val (no query lost or duplicated)",
           sorted(holdout + pool) == s._val)
    h2, _, _ = s._partition(300, 1234, 0, 5678)
    _check("meta: partition deterministic for a fixed seed", holdout == h2)
    empty, pr0, full = s._partition(0, 1234, 0, 5678)
    _check("meta: size 0 leaves the whole val split as the gate pool (run-5 parity)",
           empty == [] and pr0 == [] and full == s._val)
    # run10c cascade: three disjoint slices (meta-holdout, promote, gate pool).
    mh, pr, gp = s._partition(300, 1234, 100, 5678)
    _check("cascade: promote slice has the requested size", len(pr) == 100)
    _check("cascade: all three slices are mutually DISJOINT",
           set(mh).isdisjoint(pr) and set(mh).isdisjoint(gp)
           and set(pr).isdisjoint(gp))
    _check("cascade: the three slices partition val exactly",
           sorted(mh + pr + gp) == s._val)
    _check("cascade: meta-holdout idxs unchanged whether or not promote is carved",
           mh == holdout)


# ----------------------------------------------- failure attribution (B1/C2)

class _FakeGraph:
    def get_text(self, ids, limit=50):
        return {i: f"text-{i}" for i in ids}


def _row(recall20, recall100, retrieved=((2, 0.5),)):
    return {"query": "q", "answer_ids": [1], "retrieved": list(retrieved),
            "gold_ranks": {}, "error": None, "recall@100": recall100,
            "metrics": {"recall@20": recall20, "hit@1": 0.0, "hit@5": 0.0, "mrr": 0.2}}


def test_attribution():
    loop = FastLoop.__new__(FastLoop)
    loop._graph = _FakeGraph()
    ranking = loop._failure_record(_row(0.5, 1.0), "missed")
    _check("attribution: gold reachable in top-100 -> RANKING",
           "RANKING" in ranking["attribution"])
    generation = loop._failure_record(_row(0.0, 0.0), "missed")
    _check("attribution: gold absent from top-100 -> GENERATION",
           "GENERATION" in generation["attribution"])
    mixed = loop._failure_record(_row(0.4, 0.7), "missed")
    _check("attribution: recall@100 > recall@20 (partial) -> MIXED",
           "MIXED" in mixed["attribution"])
    empty = loop._failure_record(_row(0.0, 0.0, retrieved=()), "missed")
    _check("attribution: empty retrieval flagged", empty["empty_result"] is True)


def main():
    test_minibatch_promotion()
    test_meta_holdout_partition()
    test_attribution()
    print("\nall run-6 Phase-B tests passed")


if __name__ == "__main__":
    main()
