"""Unit tests for Phase B: per-step cost attribution from budget snapshots.

`step_cost_delta` turns two OpenAIBudget snapshots into a per-step spend dict,
routing this step's tokens to accepted-vs-rejected by the gate outcome (the
$/accepted-edit axis). Also checks that snapshot() hands out an isolated deep
copy so deltas are honest.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_cost_split
No FalkorDB / no network / no API key needed.
"""
import os
import tempfile

from graphretr_opt.env.openai_client import OpenAIBudget, step_cost_delta


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def test_step_cost_delta_split():
    prev = {"requests": 1, "tokens_in": 100, "tokens_out": 50, "usd": 0.10, "by_model": {}}
    cur = {"requests": 2, "tokens_in": 300, "tokens_out": 150, "usd": 0.25, "by_model": {}}

    acc = step_cost_delta(prev, cur, accepted=True, ceiling_usd=5.0)
    _check("accepted: tokens -> tokens_accepted",
           acc["tokens_accepted"] == 300 and acc["tokens_rejected"] == 0)
    _check("usd_step is the cumulative delta", abs(acc["usd_step"] - 0.15) < 1e-9)
    _check("usd_cumulative is the current total", acc["usd_cumulative"] == 0.25)
    _check("usd_vs_ceiling = cumulative / ceiling",
           abs(acc["usd_vs_ceiling"] - 0.05) < 1e-9)

    rej = step_cost_delta(prev, cur, accepted=False)
    _check("rejected: tokens -> tokens_rejected (spend burned on a reject)",
           rej["tokens_rejected"] == 300 and rej["tokens_accepted"] == 0)
    _check("no ceiling -> usd_vs_ceiling omitted", "usd_vs_ceiling" not in rej)


def test_snapshot_isolation_and_real_delta():
    with tempfile.TemporaryDirectory() as d:
        b = OpenAIBudget(os.path.join(d, "usage.json"), ceiling_usd=10.0)
        s0 = b.snapshot()
        b._record("gpt-4o-mini", 1_000_000, 0)          # +1M in, +$0.15
        s1 = b.snapshot()
        _check("snapshot is a copy (earlier snapshot is frozen)",
               s0["tokens_in"] == 0 and s0["usd"] == 0.0)
        delta = step_cost_delta(s0, s1, accepted=True)
        _check("snapshot delta: tokens", delta["tokens_accepted"] == 1_000_000)
        _check("snapshot delta: usd", abs(delta["usd_step"] - 0.15) < 1e-9)
        # mutating a snapshot's nested dict must not corrupt the live ledger
        b._record("gpt-4o-mini", 0, 0)
        s1["by_model"]["gpt-4o-mini"]["requests"] = 999
        _check("by_model is deep-copied",
               b.snapshot()["by_model"]["gpt-4o-mini"]["requests"] == 2)


def main():
    test_step_cost_delta_split()
    test_snapshot_isolation_and_real_delta()
    print("\nall cost_split tests passed")


if __name__ == "__main__":
    main()
