"""Phase 3 -- re-export a campaign's best_search.py via the held-out bake-off.

The original run-7 export is a rotating-gate artifact ("last belt-holder"): the
monotone incumbent at the final bell, scored on whatever 200-query rotation epoch
happened to be active -- NOT the best program re-scored on one common set. This
script loads every surviving program of a finished run (accepted/*.py + seed_used.py
+ the current best_search.py), re-scores them all on the FENCED meta-holdout via
FastLoop._final_select, and rewrites best_search.py to the argmax -- backing the old
file up as best_search_gate_artifact.py and writing select_holdout.json for audit.

It reproduces the run's meta-holdout by overlaying the run's resolved_config.yaml
(meta_holdout_size / meta_seed / gate_*), so the holdout and the blend value match
exactly what the run fenced off and gated on.

LIVE COST: this boots FalkorDB + the OpenAI gateway and re-runs the (possibly
rerank-heavy) programs over the meta-holdout (~k members x 300 queries; ~6 x 300
for run-7). Budget ~$1-2 and up to ~1-2 h wall at ~3-9 s/query. It respects
openai_budget_usd; if the ceiling is hit mid-bake-off it falls back to the best
scored so far (and logs it). Subsample with select_holdout_n (or campaign.yaml) to
cap cost/latency.

After it rewrites best_search.py, run the locked-test comparison (same 400 queries
as the lower-bound run, for apples-to-apples):

    PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli final \
        --campaign-name run7 --strategy reasoning_first_v6 --test-n 400

Run:
    PYTHONPATH=$PWD/src .venv/bin/python scripts/reexport_run7.py \
        [--campaign-name run7] [--strategy reasoning_first_v6] [--dry-run]
"""
import argparse
import glob
import os
import shutil
import types

import yaml

from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.campaign import Campaign
from graphretr_opt.config import load_config, load_strategy
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.reward.objectives import MetricVector, QUALITY_KEYS

# keys pulled from the run's resolved_config.yaml so the bake-off reproduces the
# run's holdout + blend exactly (NOT machine/infra keys, which stay current).
_REPRO_KEYS = ("meta_holdout_size", "meta_seed", "gate_size", "gate_seed",
               "gate_metric", "gate_mode", "gate_blend", "strategy",
               "select_holdout_n")


class _Pool:
    """Minimal pool shim: _final_select only reads `.members[i].program`."""

    def __init__(self, programs):
        self.members = [types.SimpleNamespace(program=p) for p in programs]


def _repro_overrides(run_dir):
    """Config keys from the run's resolved_config.yaml that pin the meta-holdout
    and blend value, so re-scoring matches the original run."""
    path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.exists(path):
        print(f"[reexport] no resolved_config.yaml in {run_dir}; "
              "using current campaign.yaml (holdout may differ from the run)")
        return {}
    doc = yaml.safe_load(open(path)) or {}
    return {k: doc[k] for k in _REPRO_KEYS if k in doc}


def _load_members(run_dir, family):
    """Every surviving program of the run, de-duped by sha: accepted/*.py +
    seed_used.py + the current best_search.py. -> [(path, SearchProgram)]."""
    paths = sorted(glob.glob(os.path.join(run_dir, "accepted", "*.py")))
    for extra in ("seed_used.py", "best_search.py"):
        p = os.path.join(run_dir, extra)
        if os.path.exists(p):
            paths.append(p)
    progs, seen = [], set()
    for path in paths:
        prog = SearchProgram.from_file(path, family=family)
        if prog.sha in seen:
            continue
        seen.add(prog.sha)
        progs.append((path, prog))
    return progs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-name", default="run7")
    ap.add_argument("--strategy", default=None,
                    help="seed family the run used (overrides resolved_config.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the members + holdout, but do not score or rewrite")
    args = ap.parse_args(argv)

    base = load_config()
    run_dir = os.path.join(base.runs_dir, args.campaign_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no such run dir: {run_dir}")

    overrides = _repro_overrides(run_dir)
    if args.strategy:
        overrides["strategy"] = args.strategy
    cfg = load_config(**overrides)
    family = load_strategy(cfg).get("family", "unknown")

    members = _load_members(run_dir, family)
    if not members:
        raise SystemExit(f"no programs found under {run_dir} (accepted/, seed_used.py)")
    print(f"[reexport] {args.campaign_name}: {len(members)} unique surviving program(s) "
          f"(family={family}, strategy={cfg.strategy})")
    for path, prog in members:
        print(f"[reexport]   {prog.sha[:8]}  {os.path.relpath(path, run_dir)}")

    best_path = os.path.join(run_dir, "best_search.py")
    gate_artifact = SearchProgram.from_file(best_path, family=family)

    if args.dry_run:
        print(f"[reexport] meta_holdout_size={cfg.meta_holdout_size} "
              f"meta_seed={cfg.meta_seed} gate_mode={cfg.gate_mode} "
              f"select_holdout_n={cfg.select_holdout_n}")
        print("[reexport] dry run -- no scoring, no rewrite.")
        return

    campaign = Campaign(cfg).boot()
    if not campaign.substrate.meta_holdout_idxs:
        print("[reexport] WARNING: meta_holdout is empty for this config -- "
              "_final_select will fall back to the fixed gate (not the fenced holdout)")

    # _final_select only uses _reward / _budget / _gate / _cfg; mutator/edit_budget/
    # tracker/archive are unused, so pass None.
    loop = FastLoop(cfg, campaign.graph, campaign.sandbox, campaign.reward,
                    mutator=None, edit_budget=None, tracker=None,
                    archive=None, budget=campaign.budget)

    fn_cache = {}

    def get_fn(p):
        f = fn_cache.get(p.sha)
        if f is None:
            f = fn_cache[p.sha] = campaign.sandbox.compile(p.src)
        return f

    pool = _Pool([prog for _, prog in members])
    export_prog, export_mv, scored = loop._final_select(
        campaign.substrate, pool, gate_artifact, MetricVector(), run_dir, get_fn)

    if not scored:
        raise SystemExit("[reexport] no member scored on the holdout -- nothing rewritten")

    # Back up the rotating-gate artifact, then write the bake-off winner.
    artifact_path = os.path.join(run_dir, "best_search_gate_artifact.py")
    if not os.path.exists(artifact_path):
        shutil.copy2(best_path, artifact_path)
        print(f"[reexport] backed up gate artifact -> {os.path.relpath(artifact_path, run_dir)}")
    else:
        print(f"[reexport] {os.path.relpath(artifact_path, run_dir)} exists -- keeping it")
    export_prog.save(best_path)

    changed = export_prog.sha != gate_artifact.sha
    print("\n==== RE-EXPORT SUMMARY ====")
    print(f"gate artifact : {gate_artifact.sha[:8]}")
    print(f"holdout winner: {export_prog.sha[:8]}  "
          + ("(DIFFERENT -- artifact was not the true best)" if changed
             else "(SAME -- the artifact happened to be the true best)"))
    print(f"winner holdout: { {k: round(export_mv.get(k), 4) for k in QUALITY_KEYS} }")
    print(f"corrected best_search.py written -> {best_path}")
    print("select_holdout.json (per-member holdout scores) written next to it.")
    print("\nNow run the locked-test comparison:")
    print(f"  PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli final "
          f"--campaign-name {args.campaign_name} --strategy {cfg.strategy} --test-n 400")


if __name__ == "__main__":
    main()
