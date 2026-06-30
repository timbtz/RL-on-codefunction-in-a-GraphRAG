"""archipelago.py -- the DAG meta-orchestrator (island-model evolutionary search).

Runs many `graphretr_opt optimize` runs as an archipelago: N independent
BRANCHES each *seed-chain* (each run's bake-off champion becomes the next run's
seed via --seed-champion) until they stop improving, then a tournament MERGE
bracket combines branch champions pairwise -- each pair feeds one --merge run
that warm-starts both champions into the pool and runs in COMBINE mode -- until a
single global champion remains. The cross-run lineage is recorded as a real DAG
artifact (archipelago_graph.json + graph.dot).

Sequential by design: one FalkorDB (port 6380) and one runs/llm_cache.json are
shared, so runs NEVER overlap -- each is a blocking subprocess. Artifacts (not
log text) are the source of truth: convergence keys off select_holdout.json +
lineage.jsonl, which are block-buffer-safe (the runs use `python -u`).

House style: print("[archipelago] ...", flush=True); no logging library.
"""
import json
import os
import subprocess
import sys

import yaml

from .config import load_config

_SPEC_DEFAULTS = {
    "campaign": "archipelago",
    "branches": 5,
    "seed": "base",                 # 'base' | list of per-branch champion paths
    "strategy_per_branch": [],      # optional: per-branch --strategy (different directions)
    "run": {"steps": 20, "checkpoint_every": 5},
    "converge": {"metric": "holdout_value", "margin": 0.01, "max_runs_per_branch": 4},
    "merge": {"policy": "tournament"},
    "execution": {"mode": "sequential", "budget_usd": 25.0},
}


# --------------------------------------------------------------- spec loading

def load_spec(path) -> dict:
    """Read the YAML campaign spec and fill defaults. Validates the few hard
    invariants (sequential-only; >=1 branch)."""
    doc = yaml.safe_load(open(path)) or {}
    spec = {**_SPEC_DEFAULTS, **doc}
    # merge nested dicts so a partial `run:`/`converge:` block keeps its siblings
    for k in ("run", "converge", "merge", "execution"):
        spec[k] = {**_SPEC_DEFAULTS[k], **(doc.get(k) or {})}
    if spec["execution"]["mode"] != "sequential":
        raise ValueError("archipelago: only execution.mode=sequential is supported "
                         "(shared FalkorDB + llm_cache)")
    if int(spec["branches"]) < 1:
        raise ValueError("archipelago: branches must be >= 1")
    return spec


# ------------------------------------------------------- convergence signal

def beat_seed(run_dir, margin):
    """The convergence recipe: did this run's exported champion beat its own seed
    on the disjoint holdout bake-off by `margin`? Returns
    (beat, champ_h, seed_h, champ_sha). `beat` is False (treat as not-beaten)
    whenever the seed is crash-excluded from the bake-off or the export did not
    move off the seed -- mirrors the run13 no-improvement case. Reads ONLY the
    JSON artifacts (lineage.jsonl line 1 parent_sha == the run's seed)."""
    lineage = os.path.join(run_dir, "lineage.jsonl")
    audit_path = os.path.join(run_dir, "select_holdout.json")
    if not (os.path.exists(lineage) and os.path.exists(audit_path)):
        return (False, None, None, None)
    with open(lineage) as f:
        first = f.readline()
    if not first.strip():
        return (False, None, None, None)
    seed_sha = json.loads(first).get("parent_sha")
    audit = json.load(open(audit_path))
    by_sha = {m["sha"]: m for m in audit.get("members", []) if not m.get("crashed")}
    exported = audit.get("exported_sha")
    champ = by_sha.get(exported)
    if champ is None:
        return (False, None, None, exported)
    champ_h = champ.get("holdout_value")
    seed_h = by_sha.get(seed_sha, {}).get("holdout_value")
    beat = (seed_h is not None and champ_h is not None
            and champ_h > seed_h + margin and exported != seed_sha)
    return (beat, champ_h, seed_h, exported)


# --------------------------------------------------------------- budget guard

def _read_usd(cfg) -> float:
    path = os.path.join(cfg.runs_dir, "openai_usage.json")
    if not os.path.exists(path):
        return 0.0
    try:
        return float(json.load(open(path)).get("usd", 0.0))
    except (ValueError, OSError):
        return 0.0


def budget_ok(spec, ledger, cfg) -> bool:
    """True while the campaign's INCREMENTAL spend (current usd minus the
    snapshot taken at campaign start, stored in the ledger so resume keeps the
    same baseline) is under execution.budget_usd."""
    cap = float(spec["execution"]["budget_usd"])
    spent = _read_usd(cfg) - float(ledger.get("usd_at_start", 0.0))
    return spent < cap


# ----------------------------------------------------------- subprocess driver

def _subprocess_runner(cfg, name, steps, *, seed_champion=None, merge_seeds=None,
                       strategy=None, checkpoint_every=5):
    """Launch ONE blocking `optimize` run (python -u, artifacts-as-truth). cwd is
    the optimizer root with src on PYTHONPATH, mirroring run3_launch.sh. Raises
    on a non-zero exit so the caller treats it as a failed (un-chainable) run."""
    cmd = [sys.executable, "-u", "-m", "graphretr_opt.cli", "optimize",
           "--campaign-name", name, "--steps", str(steps)]
    if checkpoint_every:
        cmd += ["--checkpoint-every", str(checkpoint_every)]
    if seed_champion:
        cmd += ["--seed-champion", seed_champion]
    if strategy:
        cmd += ["--strategy", strategy]
    if merge_seeds:
        cmd += ["--merge", "--merge-seed", *merge_seeds]
    env = {**os.environ, "PYTHONPATH": cfg.opt_src_abs}
    print(f"[archipelago] launch {name}: {' '.join(cmd[3:])}", flush=True)
    proc = subprocess.run(cmd, cwd=cfg.root, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"run {name} exited {proc.returncode}")


# --------------------------------------------------------------- DAG artifact

def _champ_path(cfg, run_id):
    return os.path.join(cfg.runs_dir, run_id, "best_search.py")


def write_dag(ledger, out_dir):
    """Persist the lineage ledger (archipelago_graph.json) and re-emit Graphviz
    (graph.dot). Called after every run so a kill leaves a current DAG."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "archipelago_graph.json")
    with open(json_path, "w") as f:
        json.dump(ledger, f, indent=1)
    dot_path = os.path.join(out_dir, "graph.dot")
    lines = ["digraph archipelago {", "  rankdir=LR;",
             '  node [shape=box, style=rounded];']
    champ = ledger.get("champion")
    for n in ledger["nodes"]:
        hv = n.get("holdout_value")
        label = n["id"] + (f"\\nh={hv:.4f}" if isinstance(hv, (int, float)) else "")
        attrs = [f'label="{label}"']
        if n.get("kind") == "merge":
            attrs.append('style="rounded,filled"')
            attrs.append('fillcolor="#e6f0ff"')
        if n["id"] == champ:
            attrs.append('color="#1a7f37"')
            attrs.append('penwidth=3')
        lines.append(f'  "{n["id"]}" [{", ".join(attrs)}];')
    for a, b in ledger["edges"]:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    with open(dot_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, dot_path


def _record(ledger, *, run_id, kind, branch, seed_from, run_dir, margin,
            planned=False):
    """Append (or refresh) a node + its seed edges in the ledger. Reads the
    convergence metrics from the run's artifacts unless `planned` (dry-run)."""
    beat = champ_h = seed_h = champ_sha = None
    if not planned:
        beat, champ_h, seed_h, champ_sha = beat_seed(run_dir, margin)
    node = {"id": run_id, "kind": kind, "branch": branch,
            "seed_from": list(seed_from), "champion_sha": champ_sha,
            "holdout_value": champ_h, "beat_seed": bool(beat) if not planned else None,
            "planned": planned}
    ledger["nodes"] = [n for n in ledger["nodes"] if n["id"] != run_id] + [node]
    existing = {tuple(e) for e in ledger["edges"]}
    for parent in seed_from:
        if (parent, run_id) not in existing:
            ledger["edges"].append([parent, run_id])
    return node


# ------------------------------------------------------------- run scheduling

def _has_result(cfg, run_id):
    return os.path.exists(os.path.join(cfg.runs_dir, run_id, "select_holdout.json"))


def _run_and_record(cfg, spec, ledger, camp_dir, *, run_id, kind, branch,
                    seed_from, seed_champion=None, merge_seeds=None, strategy=None,
                    dry_run=False, runner=None):
    """Drive one run (or plan it in dry-run / skip it on resume), record its node,
    and re-emit the DAG. Returns the node dict, or None if the run failed."""
    margin = float(spec["converge"]["margin"])
    run_dir = os.path.join(cfg.runs_dir, run_id)
    if dry_run:
        node = _record(ledger, run_id=run_id, kind=kind, branch=branch,
                       seed_from=seed_from, run_dir=run_dir, margin=margin,
                       planned=True)
        write_dag(ledger, camp_dir)
        print(f"[archipelago] PLAN {run_id} (kind={kind}, branch={branch}, "
              f"seed_from={list(seed_from)})", flush=True)
        return node
    if _has_result(cfg, run_id):
        print(f"[archipelago] resume: {run_id} already has select_holdout.json "
              f"-- skipping launch", flush=True)
    else:
        if not budget_ok(spec, ledger, cfg):
            print(f"[archipelago] budget exhausted before {run_id} -- not "
                  f"launching", flush=True)
            return None
        runner = runner or _subprocess_runner
        try:
            runner(cfg, run_id, int(spec["run"]["steps"]),
                   seed_champion=seed_champion, merge_seeds=merge_seeds,
                   strategy=strategy,
                   checkpoint_every=int(spec["run"]["checkpoint_every"]))
        except Exception as e:  # a crashed/killed run is not chainable
            print(f"[archipelago] run {run_id} FAILED: {e}", flush=True)
            return None
    if not _has_result(cfg, run_id):
        print(f"[archipelago] run {run_id} produced no select_holdout.json "
              f"-- treating as failed", flush=True)
        return None
    node = _record(ledger, run_id=run_id, kind=kind, branch=branch,
                   seed_from=seed_from, run_dir=run_dir, margin=margin)
    write_dag(ledger, camp_dir)
    print(f"[archipelago] done {run_id}: holdout={node['holdout_value']} "
          f"beat_seed={node['beat_seed']}", flush=True)
    return node


def run_branch(cfg, spec, ledger, camp_dir, branch, seed0, *, dry_run=False,
               runner=None):
    """Seed-chain one branch: run, beat-seed test, chain from the champion until
    not-beaten or max_runs_per_branch. Returns the branch's best node."""
    max_runs = int(spec["converge"]["max_runs_per_branch"])
    strategies = spec.get("strategy_per_branch") or []
    strategy = strategies[branch] if branch < len(strategies) else None
    if dry_run:                       # one nominal run per branch (chain unknown)
        max_runs = 1
    seed_champion = None if seed0 in (None, "base") else seed0
    best = None
    parents = list(seed0["seed_from"]) if isinstance(seed0, dict) else []
    for r in range(max_runs):
        run_id = f"{spec['campaign']}_b{branch}_r{r}"
        seed_from = parents if r == 0 else [f"{spec['campaign']}_b{branch}_r{r-1}"]
        node = _run_and_record(cfg, spec, ledger, camp_dir, run_id=run_id,
                               kind="chain", branch=branch, seed_from=seed_from,
                               seed_champion=seed_champion, strategy=strategy,
                               dry_run=dry_run, runner=runner)
        if node is None:
            break
        if best is None or _hv(node) >= _hv(best):
            best = node
        # chain off this run's champion next iteration
        seed_champion = _champ_path(cfg, run_id)
        if dry_run:
            break
        if not node.get("beat_seed"):
            print(f"[archipelago] branch {branch}: {run_id} did not beat its seed "
                  f"-- stopping chain", flush=True)
            break
    return best


def _hv(node):
    v = node.get("holdout_value") if node else None
    return v if isinstance(v, (int, float)) else float("-inf")


def _pairs(items):
    """Tournament pairing for one round: [(0,1),(2,3),...] + an odd bye."""
    pairs = [(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
    bye = items[-1] if len(items) % 2 else None
    return pairs, bye


def merge_bracket(cfg, spec, ledger, camp_dir, champions, *, dry_run=False,
                  runner=None):
    """Pairwise tournament: each pair feeds one --merge run warm-started with both
    champions; the run's bake-off exports the best of {parentA, parentB, combos},
    so a merge that finds nothing new simply passes the better parent forward.
    Recurse on winners; an odd champion byes. Returns the single global champion."""
    current = [c for c in champions if c is not None]
    rnd = 0
    while len(current) > 1:
        pairs, bye = _pairs(current)
        winners = []
        for i, (a, b) in enumerate(pairs):
            run_id = f"{spec['campaign']}_m{rnd}_{i}"
            merge_seeds = None if dry_run else [_champ_path(cfg, a["id"]),
                                                _champ_path(cfg, b["id"])]
            node = _run_and_record(cfg, spec, ledger, camp_dir, run_id=run_id,
                                   kind="merge", branch=-1,
                                   seed_from=[a["id"], b["id"]],
                                   merge_seeds=merge_seeds, dry_run=dry_run,
                                   runner=runner)
            # a failed merge run still converges the bracket: keep the better parent
            winners.append(node if node is not None else
                           (a if _hv(a) >= _hv(b) else b))
        if bye is not None:
            print(f"[archipelago] merge round {rnd}: {bye['id']} byes", flush=True)
            winners.append(bye)
        current = winners
        rnd += 1
    return current[0] if current else None


# --------------------------------------------------------------- entrypoint

def _new_ledger(campaign):
    return {"campaign": campaign, "usd_at_start": None, "nodes": [], "edges": [],
            "champion": None}


def _load_ledger(path, campaign):
    if os.path.exists(path):
        led = json.load(open(path))
        led.setdefault("usd_at_start", None)
        return led
    return _new_ledger(campaign)


def run_campaign(spec_path, dry_run=False, resume=False, runner=None):
    """Top-level driver: load spec, run/plan each branch sequentially, merge the
    branch champions into one, finalize the DAG. Resumable (skips runs that
    already have select_holdout.json); budget-capped."""
    spec = load_spec(spec_path)
    cfg = load_config()
    campaign = spec["campaign"]
    camp_dir = os.path.join(cfg.runs_dir, campaign)
    os.makedirs(camp_dir, exist_ok=True)
    ledger_path = os.path.join(camp_dir, "archipelago_graph.json")
    ledger = _load_ledger(ledger_path, campaign) if resume else _new_ledger(campaign)
    if ledger.get("usd_at_start") is None:
        ledger["usd_at_start"] = _read_usd(cfg)
    n_branches = int(spec["branches"])
    seeds = spec.get("seed")
    print(f"[archipelago] campaign '{campaign}': {n_branches} branches, "
          f"steps={spec['run']['steps']}, margin={spec['converge']['margin']}, "
          f"budget=${spec['execution']['budget_usd']} "
          f"{'[DRY-RUN]' if dry_run else ''}", flush=True)

    champions = []
    for b in range(n_branches):
        if isinstance(seeds, list):
            seed0 = seeds[b] if b < len(seeds) else "base"
        else:
            seed0 = "base"
        if not dry_run and not budget_ok(spec, ledger, cfg):
            print(f"[archipelago] budget exhausted -- stopping after {b} branches",
                  flush=True)
            break
        champ = run_branch(cfg, spec, ledger, camp_dir, b, seed0,
                           dry_run=dry_run, runner=runner)
        if champ is not None:
            champions.append(champ)

    global_champ = None
    if champions:
        if not dry_run and not budget_ok(spec, ledger, cfg):
            print("[archipelago] budget exhausted -- skipping merge bracket; "
                  "global champion = best branch", flush=True)
            global_champ = max(champions, key=_hv)
        else:
            global_champ = merge_bracket(cfg, spec, ledger, camp_dir, champions,
                                         dry_run=dry_run, runner=runner)
    ledger["champion"] = global_champ["id"] if global_champ else None
    json_path, dot_path = write_dag(ledger, camp_dir)
    print(f"[archipelago] DONE. champion={ledger['champion']} "
          f"-> {json_path}", flush=True)
    print(f"[archipelago] render: dot -Tsvg {dot_path} -o "
          f"{os.path.join(camp_dir, 'graph.svg')}", flush=True)
    return ledger
