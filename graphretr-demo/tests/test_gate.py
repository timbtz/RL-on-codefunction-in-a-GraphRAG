"""Unit tests for the accept Gate -- blend (anti-overfit) + strict + rotation.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_gate
No FalkorDB / no network needed.
"""
from graphretr_opt.optimizer.gate import Gate


class _MV:
    """Minimal stand-in for a MetricVector (Gate only needs .get + .crashed)."""

    def __init__(self, recall, mrr, hit1=0.0, crashed=False):
        self._q = {"recall@20": recall, "mrr": mrr, "hit@1": hit1}
        self.crashed = crashed

    def get(self, k):
        return self._q[k]


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def main():
    blend = {"recall@20": 0.6, "mrr": 0.4}
    g = Gate(mode="blend", blend=blend)

    inc = _MV(recall=0.40, mrr=0.20)

    # recall up AND mrr up -> accept
    _check("blend: both up accepts", g.accept(_MV(0.42, 0.21), inc))

    # the run1/run2 failure mode: recall pumped, mrr tanked. Net composite must
    # decide. +0.02 recall (*0.6 = +0.012) but -0.05 mrr (*0.4 = -0.020) -> reject.
    _check("blend: recall-up/mrr-crash rejects",
           not g.accept(_MV(0.42, 0.15), inc))

    # recall up enough to outweigh a small mrr dip -> accept
    _check("blend: recall up outweighs tiny mrr dip",
           g.accept(_MV(0.46, 0.19), inc))

    # tie on composite -> reject (strict >)
    _check("blend: exact tie rejects", not g.accept(_MV(0.40, 0.20), inc))

    # crashed candidate never accepted, however good the numbers
    _check("blend: crashed rejects", not g.accept(_MV(0.99, 0.99, crashed=True), inc))

    # run-4 Hit@1 blend: weight on recall@20:0.4, mrr:0.3, hit@1:0.3. A candidate
    # that lifts ranking (hit@1 + mrr) at a tiny recall cost should now accept,
    # where the old recall-heavy blend would have rejected it.
    g4 = Gate(mode="blend", blend={"recall@20": 0.4, "mrr": 0.3, "hit@1": 0.3})
    inc4 = _MV(recall=0.42, mrr=0.20, hit1=0.10)
    _check("hit@1 blend: ranking gain at tiny recall cost accepts",
           g4.accept(_MV(0.41, 0.28, hit1=0.20), inc4))
    _check("hit@1 blend: recall-only pump with ranking flat rejects",
           not g4.accept(_MV(0.44, 0.18, hit1=0.08), inc4))

    # strict mode still works and ignores mrr
    gs = Gate(mode="strict", metric="recall@20")
    _check("strict: recall up accepts even if mrr down",
           gs.accept(_MV(0.41, 0.10), inc))
    _check("strict: recall tie rejects", not gs.accept(_MV(0.40, 0.99), inc))

    print("\nall gate tests passed")


if __name__ == "__main__":
    main()
