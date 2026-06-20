"""Unit tests for Phase A: the instance-wise Pareto CandidatePool -- sole-best
counting, admission (non-dominated OR sole-best), dominated-pruning in parent
selection, proportional-win sampling determinism, and cap eviction that keeps
the frontier.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_pool_select
No FalkorDB / no network needed.
"""
import random

from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.optimizer.pool import CandidatePool, _sole_best_counts
from graphretr_opt.reward.objectives import MetricVector


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _prog(tag):
    return SearchProgram(f"def search(q, G):\n    return {{{tag}: 1.0}}\n", family="t")


def _mv(primary, complexity=10.0, per_query=None, latency=0.0, db=0.0):
    return MetricVector(
        quality={"recall@20": primary, "hit@1": 0.0, "hit@5": 0.0, "mrr": primary},
        latency_s=latency, db_load=db, code_complexity=complexity,
        per_query=per_query or {})


def _member(pool, prog):
    return next(m for m in pool.members if m.sha == prog.sha)


def test_sole_best_counts():
    a = _mv(0.5, per_query={1: {"mrr": 0.9}, 2: {"mrr": 0.1}, 3: {"mrr": 0.5}})
    b = _mv(0.4, per_query={1: {"mrr": 0.2}, 2: {"mrr": 0.8}, 3: {"mrr": 0.5}})
    c = _mv(0.3, per_query={1: {"mrr": 0.0}, 2: {"mrr": 0.0}, 3: {"mrr": 0.0}})

    class _M:
        def __init__(self, mv):
            self.metrics = mv

    counts = _sole_best_counts([_M(a), _M(b), _M(c)])
    _check("sole-best: A wins q1, B wins q2", counts[0] == 1 and counts[1] == 1)
    _check("sole-best: tie on q3 awards nobody", sum(counts) == 2)
    _check("sole-best: all-zero query awards nobody (C wins 0)", counts[2] == 0)


def test_admission_rules():
    pool = CandidatePool(cap=24)
    pa = _prog("1")
    fg, adm = pool.consider(pa, _mv(0.40, per_query={1: {"mrr": 0.4}}))
    _check("admit: first member grows the frontier", fg and adm)

    # strictly better (dominates pa) -> new frontier point
    pb = _prog("2")
    fg, adm = pool.consider(pb, _mv(0.50, per_query={1: {"mrr": 0.5}}))
    _check("admit: dominating child grows the frontier", fg and adm)
    _check("admit: dominated parent dropped from pool", pa.sha not in pool.shas())

    # dominated on aggregate, but sole-best on a NEW query -> admitted, no frontier
    pc = _prog("3")
    fg, adm = pool.consider(pc, _mv(0.30, per_query={2: {"mrr": 0.99}}))
    _check("admit: dominated specialist admitted (not frontier)", adm and not fg)

    # dominated on aggregate AND sole-best on nothing -> rejected
    pd = _prog("4")
    fg, adm = pool.consider(pd, _mv(0.20, per_query={1: {"mrr": 0.0}}))
    _check("reject: dominated non-specialist not admitted", (not adm) and (not fg))


def test_select_prunes_dominated_winners():
    pool = CandidatePool(cap=24)
    # A dominates C on aggregate; both win a query. C must be pruned as a parent.
    pa, pb, pc = _prog("a"), _prog("b"), _prog("c")
    pool.consider(pa, _mv(0.50, complexity=10, per_query={1: {"mrr": 0.9}}))
    pool.consider(pb, _mv(0.40, complexity=5, per_query={2: {"mrr": 0.9}}))  # frontier (better complexity)
    pool.consider(pc, _mv(0.45, complexity=10, per_query={3: {"mrr": 0.9}}))  # dominated by A
    _check("setup: C admitted as specialist", pc.sha in pool.shas())
    picks = set()
    for i in range(200):
        picks.add(pool.select_parent(random.Random(i), discount=False).sha)
    _check("select: dominated winner C never chosen as parent", pc.sha not in picks)
    _check("select: both frontier winners A and B reachable",
           pa.sha in picks and pb.sha in picks)


def test_select_proportional_and_deterministic():
    pa, pb = _prog("a"), _prog("b")

    def fresh():
        p = CandidatePool(cap=24)
        # A is sole-best on 3 queries, B on 1; neither dominates the other.
        p.consider(pa, _mv(0.50, complexity=10,
                           per_query={1: {"mrr": .9}, 2: {"mrr": .9}, 3: {"mrr": .9},
                                      9: {"mrr": .1}}))
        p.consider(pb, _mv(0.40, complexity=5,
                           per_query={1: {"mrr": .1}, 2: {"mrr": .1}, 3: {"mrr": .1},
                                      9: {"mrr": .9}}))
        return p

    s1 = fresh().select_parent(random.Random(7), discount=False).sha
    s2 = fresh().select_parent(random.Random(7), discount=False).sha
    _check("select: same seed -> same parent (deterministic)", s1 == s2)

    counts = {pa.sha: 0, pb.sha: 0}
    for i in range(400):
        counts[fresh().select_parent(random.Random(1000 + i), discount=False).sha] += 1
    _check("select: 3:1 win ratio makes A the more frequent parent",
           counts[pa.sha] > counts[pb.sha] * 1.8)


def test_cap_eviction_keeps_frontier():
    pool = CandidatePool(cap=2)
    pa, pb, pc = _prog("a"), _prog("b"), _prog("c")
    pool.consider(pa, _mv(0.50, complexity=10, per_query={1: {"mrr": 0.9}}))
    pool.consider(pb, _mv(0.40, complexity=5, per_query={2: {"mrr": 0.9}}))  # frontier
    pool.consider(pc, _mv(0.45, complexity=10, per_query={3: {"mrr": 0.9}}))  # dominated specialist
    _check("evict: pool capped at 2", len(pool) == 2)
    _check("evict: both frontier members retained",
           pa.sha in pool.shas() and pb.sha in pool.shas())
    _check("evict: dominated specialist C evicted", pc.sha not in pool.shas())


def main():
    test_sole_best_counts()
    test_admission_rules()
    test_select_prunes_dominated_winners()
    test_select_proportional_and_deterministic()
    test_cap_eviction_keeps_frontier()
    print("\nall pool_select tests passed")


if __name__ == "__main__":
    main()
