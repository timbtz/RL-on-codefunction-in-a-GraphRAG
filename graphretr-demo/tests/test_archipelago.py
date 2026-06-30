"""Unit tests for the archipelago orchestrator's PURE logic (no optimizer
launched). Fake select_holdout.json / lineage.jsonl are written to a tmp runs
dir and a fake runner is injected, so the convergence recipe, tournament
pairing, DAG artifact, budget guard and dry-run plan are all exercised
infra-free.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_archipelago
No FalkorDB / no network needed.
"""
import json
import os
import tempfile
import types

from graphretr_opt import archipelago as A


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _write_run(run_dir, *, seed_sha, exported_sha, members):
    """Write the two convergence artifacts a real run produces.
    members: list of (sha, holdout_value, crashed)."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "lineage.jsonl"), "w") as f:
        f.write(json.dumps({"step": 0, "parent_sha": seed_sha,
                            "child_sha": "x"}) + "\n")
    audit = {"exported_sha": exported_sha,
             "members": [{"sha": s, "holdout_value": h, "crashed": c}
                         for (s, h, c) in members]}
    with open(os.path.join(run_dir, "select_holdout.json"), "w") as f:
        json.dump(audit, f)


# ----------------------------------------------------------- beat_seed recipe

def test_beat_seed_false_when_export_is_seed():
    with tempfile.TemporaryDirectory() as d:
        # run13 case: the export did not move off the seed -> not beaten.
        _write_run(d, seed_sha="S", exported_sha="S",
                   members=[("S", 0.40, False)])
        beat, champ_h, seed_h, sha = A.beat_seed(d, margin=0.01)
        _check("export==seed -> beat_seed False", beat is False)
        _check("seed holdout reported", abs(seed_h - 0.40) < 1e-9)


def test_beat_seed_true_when_champion_beats_seed():
    with tempfile.TemporaryDirectory() as d:
        _write_run(d, seed_sha="S", exported_sha="C",
                   members=[("S", 0.40, False), ("C", 0.45, False)])
        beat, champ_h, seed_h, sha = A.beat_seed(d, margin=0.01)
        _check("champion beats seed by > margin -> True", beat is True)
        _check("champion sha returned", sha == "C")
        _check("champion holdout reported", abs(champ_h - 0.45) < 1e-9)


def test_beat_seed_false_within_margin():
    with tempfile.TemporaryDirectory() as d:
        _write_run(d, seed_sha="S", exported_sha="C",
                   members=[("S", 0.40, False), ("C", 0.405, False)])
        beat, *_ = A.beat_seed(d, margin=0.01)
        _check("gain within margin -> not beaten", beat is False)


def test_beat_seed_false_when_seed_crash_excluded():
    with tempfile.TemporaryDirectory() as d:
        # seed crashed out of the bake-off -> absent from members -> not beaten.
        _write_run(d, seed_sha="S", exported_sha="C",
                   members=[("S", 0.0, True), ("C", 0.45, False)])
        beat, champ_h, seed_h, _ = A.beat_seed(d, margin=0.01)
        _check("seed crash-excluded -> beat_seed False", beat is False)
        _check("seed holdout is None (not in bake-off)", seed_h is None)


def test_beat_seed_false_when_no_artifacts():
    with tempfile.TemporaryDirectory() as d:
        beat, *_ = A.beat_seed(d, margin=0.01)
        _check("missing artifacts -> not beaten (no crash)", beat is False)


# --------------------------------------------------------- tournament pairing

def test_pairs_even():
    pairs, bye = A._pairs([1, 2, 3, 4])
    _check("even: (1,2)(3,4)", pairs == [(1, 2), (3, 4)])
    _check("even: no bye", bye is None)


def test_pairs_odd():
    pairs, bye = A._pairs([1, 2, 3])
    _check("odd: (1,2)", pairs == [(1, 2)])
    _check("odd: 3 byes", bye == 3)


# --------------------------------------------------------------- DAG artifact

def test_write_dag_shape():
    ledger = {"campaign": "c", "usd_at_start": 0.0,
              "nodes": [
                  {"id": "c_b0_r0", "kind": "chain", "branch": 0,
                   "seed_from": [], "champion_sha": "C0",
                   "holdout_value": 0.41, "beat_seed": True, "planned": False},
                  {"id": "c_b1_r0", "kind": "chain", "branch": 1,
                   "seed_from": [], "champion_sha": "C1",
                   "holdout_value": 0.39, "beat_seed": True, "planned": False},
                  {"id": "c_m0_0", "kind": "merge", "branch": -1,
                   "seed_from": ["c_b0_r0", "c_b1_r0"], "champion_sha": "M",
                   "holdout_value": 0.43, "beat_seed": None, "planned": False},
              ],
              "edges": [["c_b0_r0", "c_m0_0"], ["c_b1_r0", "c_m0_0"]],
              "champion": "c_m0_0"}
    with tempfile.TemporaryDirectory() as d:
        json_path, dot_path = A.write_dag(ledger, d)
        reloaded = json.load(open(json_path))
        _check("graph.json round-trips nodes", len(reloaded["nodes"]) == 3)
        _check("graph.json records champion", reloaded["champion"] == "c_m0_0")
        dot = open(dot_path).read()
        _check("dot is a digraph", dot.startswith("digraph archipelago {"))
        _check("dot has the merge node", '"c_m0_0"' in dot)
        _check("dot has a lineage edge", '"c_b0_r0" -> "c_m0_0";' in dot)
        _check("dot labels the holdout value", "h=0.4300" in dot)


# ------------------------------------------------- dry-run plans, no launches

def test_dry_run_plans_and_launches_nothing():
    with tempfile.TemporaryDirectory() as d:
        spec_path = os.path.join(d, "spec.yaml")
        with open(spec_path, "w") as f:
            f.write("campaign: t\nbranches: 3\nrun:\n  steps: 5\n"
                    "execution:\n  mode: sequential\n  budget_usd: 10\n")

        launched = []

        def _spy_runner(*a, **k):
            launched.append((a, k))

        # point runs_dir at the tmp dir via a patched load_config
        real_load = A.load_config

        def _fake_load(**kw):
            cfg = real_load(**kw)
            return types.SimpleNamespace(
                runs_dir=os.path.join(d, "runs"), root=d,
                opt_src_abs=cfg.opt_src_abs)
        A.load_config = _fake_load
        try:
            ledger = A.run_campaign(spec_path, dry_run=True, runner=_spy_runner)
        finally:
            A.load_config = real_load

        _check("dry-run launches NOTHING", launched == [])
        chain = [n for n in ledger["nodes"] if n["kind"] == "chain"]
        merge = [n for n in ledger["nodes"] if n["kind"] == "merge"]
        _check("plans one node per branch (3)", len(chain) == 3)
        _check("plans the merge bracket (3 champs -> 2 merges)", len(merge) == 2)
        _check("all planned nodes flagged planned",
               all(n["planned"] for n in ledger["nodes"]))
        _check("DAG written to runs/<campaign>", os.path.exists(
            os.path.join(d, "runs", "t", "archipelago_graph.json")))


# -------------------------------------------- real(ish) drive with a fake runner

def test_run_campaign_with_fake_runner_chains_and_merges():
    """A 2-branch campaign driven by a fake runner that fabricates each run's
    artifacts: branch chains stop after one run (champion does not beat its own
    seed on run 1), then the two branch champions merge into one."""
    with tempfile.TemporaryDirectory() as d:
        runs = os.path.join(d, "runs")
        spec_path = os.path.join(d, "spec.yaml")
        with open(spec_path, "w") as f:
            f.write("campaign: t\nbranches: 2\nrun:\n  steps: 5\n"
                    "converge:\n  margin: 0.01\n  max_runs_per_branch: 2\n"
                    "execution:\n  mode: sequential\n  budget_usd: 100\n")

        # each fake run: the exported champion BEATS its seed on run r0 (so the
        # chain advances once) but on r1 the export == seed (chain stops).
        def _fake_runner(cfg, name, steps, *, seed_champion=None,
                         merge_seeds=None, strategy=None, checkpoint_every=5):
            run_dir = os.path.join(runs, name)
            if name.endswith("_r0"):
                _write_run(run_dir, seed_sha="SEED", exported_sha=name + "_C",
                           members=[("SEED", 0.30, False),
                                    (name + "_C", 0.42, False)])
            elif "_m" in name:  # merge run: exports a fresh combined champion
                _write_run(run_dir, seed_sha="SEED", exported_sha=name + "_C",
                           members=[("SEED", 0.30, False),
                                    (name + "_C", 0.50, False)])
            else:               # r1: no improvement -> chain stops
                _write_run(run_dir, seed_sha="SEED", exported_sha="SEED",
                           members=[("SEED", 0.42, False)])
            # the champion file the next stage seeds/merges from
            with open(os.path.join(run_dir, "best_search.py"), "w") as fh:
                fh.write("# champion\n")

        real_load = A.load_config

        def _fake_load(**kw):
            cfg = real_load(**kw)
            return types.SimpleNamespace(runs_dir=runs, root=d,
                                         opt_src_abs=cfg.opt_src_abs)
        A.load_config = _fake_load
        try:
            ledger = A.run_campaign(spec_path, runner=_fake_runner)
        finally:
            A.load_config = real_load

        chain = [n for n in ledger["nodes"] if n["kind"] == "chain"]
        merge = [n for n in ledger["nodes"] if n["kind"] == "merge"]
        _check("each branch chained exactly 2 runs (r0 beat, r1 stopped)",
               len([n for n in chain if n["branch"] == 0]) == 2)
        _check("one merge run combined the two branch champions", len(merge) == 1)
        _check("global champion is the merge winner",
               ledger["champion"] == merge[0]["id"])
        _check("merge node seeds from both branch champions",
               set(merge[0]["seed_from"]) == {"t_b0_r1", "t_b1_r1"})


def _run():
    test_beat_seed_false_when_export_is_seed()
    test_beat_seed_true_when_champion_beats_seed()
    test_beat_seed_false_within_margin()
    test_beat_seed_false_when_seed_crash_excluded()
    test_beat_seed_false_when_no_artifacts()
    test_pairs_even()
    test_pairs_odd()
    test_write_dag_shape()
    test_dry_run_plans_and_launches_nothing()
    test_run_campaign_with_fake_runner_chains_and_merges()
    print("all archipelago tests passed")


if __name__ == "__main__":
    _run()
