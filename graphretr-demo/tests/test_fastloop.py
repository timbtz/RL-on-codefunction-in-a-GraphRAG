"""Unit tests for FastLoop._reflect / _failure_record on synthetic rows: the
two-bucket split (missed vs misranked), the success-path winners, and the
μ_f signal (missed-gold texts + the non-gold nodes that out-ranked the gold,
with the program's scores + gold's actual rank).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_fastloop
No FalkorDB / no network needed (a fake graph supplies node texts).
"""
from graphretr_opt.optimizer.fast_loop import FastLoop


class _FakeGraph:
    def get_text(self, ids, limit=50):
        return {i: f"text-of-{i}" for i in ids}


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _row(query, recall, hit1, mrr, answer_ids, retrieved, gold_ranks, error=None):
    return {"query": query,
            "metrics": {"recall@20": recall, "hit@1": hit1, "hit@5": 0.0, "mrr": mrr},
            "answer_ids": answer_ids, "retrieved": retrieved,
            "gold_ranks": gold_ranks, "error": error}


def main():
    loop = FastLoop.__new__(FastLoop)   # bypass __init__ (no cfg/graph needed)
    loop._graph = _FakeGraph()

    rows = [
        # missed: gold 2 absent from top-20; gold 1 retrieved at rank 2
        _row("q-missed", 0.5, 0.0, 0.20, [1, 2],
             [(9, 0.8), (8, 0.7), (1, 0.3)], {1: (2, 0.3)}),
        # misranked: full recall but gold 5 sits at rank 1 behind non-gold 7
        _row("q-misranked", 1.0, 0.0, 0.50, [5],
             [(7, 0.9), (5, 0.4)], {5: (1, 0.4)}),
        # winner: hit@1 perfect -> protected
        _row("q-win", 1.0, 1.0, 1.0, [3], [(3, 0.99)], {3: (0, 0.99)}),
    ]

    failures, wins = loop._reflect(rows, n=2)

    buckets = {f["bucket"] for f in failures}
    _check("both buckets surfaced", buckets == {"missed", "misranked"})

    missed = next(f for f in failures if f["bucket"] == "missed")
    _check("missed-gold text shown", missed["missed_gold"] == [(2, "text-of-2")])
    _check("out-ranking non-gold shown with score",
           (9, 0.8, "text-of-9") in missed["top_wrong"])
    _check("retrieved gold rank shown", (2, 0.3, 1) in missed["gold_ranks"])

    misr = next(f for f in failures if f["bucket"] == "misranked")
    _check("misranked names the non-gold beating gold",
           misr["top_wrong"][0][0] == 7)
    _check("misranked has no missing gold", misr["missed_gold"] == [])

    _check("winner surfaced for do-not-regress",
           len(wins) == 1 and wins[0]["query"] == "q-win")
    _check("winner carries metrics", wins[0]["metrics"]["hit@1"] == 1.0)

    print("\nall fastloop tests passed")


if __name__ == "__main__":
    main()
