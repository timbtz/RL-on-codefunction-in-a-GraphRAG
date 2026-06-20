"""Unit tests for Phase 2: the stop counter is decoupled from frontier growth.

Run-7 pathology: step 27 was a REAL admission (sole-best specialist,
frontier_grew=False). Because the stop counter keyed off frontier growth, that
admission did not reset it -- `stale` ran 6->7->8 and the campaign stopped at
29/40, throwing ~25% of the step budget while still admitting useful programs.

FastLoop._is_stop_progress(pool_on, admitted, accepted) is the decoupled signal:
the stop resets on ANY admission (frontier OR sole-best) when the pool drives the
search, else on a headline accept. Frontier growth still drives architect
escalation separately (not tested here -- that path is unchanged).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_stale_stop
No FalkorDB / no network needed.
"""
from graphretr_opt.optimizer.fast_loop import FastLoop


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _simulate(events, stop_after, pool_on=True):
    """Replicate run()'s stop-counter loop over a scripted (admitted, accepted)
    sequence. -> (stopped_at_index or None, final_counter)."""
    ctr = 0
    for i, (admitted, accepted) in enumerate(events):
        if FastLoop._is_stop_progress(pool_on, admitted, accepted):
            ctr = 0
        else:
            ctr += 1
        if stop_after and ctr >= stop_after:
            return i, ctr
    return None, ctr


def test_stop_progress_signal():
    # pool ON: any admission is progress for the stop, regardless of frontier.
    _check("sole-best admission (frontier_grew=False) resets the stop",
           FastLoop._is_stop_progress(True, admitted=True, accepted=True) is True)
    _check("non-admission is not progress",
           FastLoop._is_stop_progress(True, admitted=False, accepted=False) is False)
    # pool OFF: the headline-accept drives the stop (run-5 parity).
    _check("pool off: headline accept is progress",
           FastLoop._is_stop_progress(False, admitted=False, accepted=True) is True)
    _check("pool off: non-accept is not progress",
           FastLoop._is_stop_progress(False, admitted=False, accepted=False) is False)


def test_sole_best_admission_resets_and_run_continues():
    # The run-7 shape: a long reject streak, then a sole-best admission, then more
    # rejects. With stop_after=8 the admission must reset the counter so the run
    # does NOT stop while it is still admitting useful programs.
    reject = (False, False)
    sole_best = (True, True)      # admitted but frontier_grew=False
    events = [reject] * 7 + [sole_best] + [reject] * 7
    stopped_at, ctr = _simulate(events, stop_after=8)
    _check("run does NOT stop -- sole-best admission reset the stop counter",
           stopped_at is None)
    _check("counter after the post-admission rejects is 7 (< 8)", ctr == 7)


def test_stops_only_on_genuine_admission_drought():
    # 8 consecutive non-admissions with no reset -> stop fires at index 7.
    events = [(False, False)] * 10
    stopped_at, _ = _simulate(events, stop_after=8)
    _check("8 straight non-admissions trigger the stop", stopped_at == 7)


def test_frontier_growth_not_required_to_keep_going():
    # A pool that only ever admits specialists (frontier never grows) must keep
    # running -- this is precisely what the old frontier-keyed stop got wrong.
    specialist = (True, True)
    events = [specialist] * 20
    stopped_at, ctr = _simulate(events, stop_after=4)
    _check("specialist-only run never stops", stopped_at is None and ctr == 0)


def main():
    test_stop_progress_signal()
    test_sole_best_admission_resets_and_run_continues()
    test_stops_only_on_genuine_admission_drought()
    test_frontier_growth_not_required_to_keep_going()
    print("\nall stale_stop tests passed")


if __name__ == "__main__":
    main()
