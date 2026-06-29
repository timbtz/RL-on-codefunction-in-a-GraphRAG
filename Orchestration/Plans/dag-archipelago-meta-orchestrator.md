# Feature: DAG Archipelago Meta-Orchestrator

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files etc. This plan was built from a read-only codebase audit (4 parallel agents); every `file:line` below was verified against the current tree, but re-confirm before editing — line numbers drift.

## Feature Description

A **sequential, YAML-driven meta-orchestrator** ("DAG executor") that runs many `graphretr_opt` optimization runs as an **island-model evolutionary search**: N independent branches each *seed-chain* (each run's bake-off champion becomes the next run's seed) until they stop improving, then a **tournament merge bracket** combines branch champions pairwise — feeding two champions into one run that uses the existing COMBINE/crossover mode — recursively until a single global champion remains. The orchestrator records the cross-run lineage as a real **DAG artifact** (`graph.json` + Graphviz `graph.dot`).

Today there is **no cross-run mechanism** at all: each run starts from whatever is on disk in `starksearch/src/stark_search/stark_graph_search_service.py`, and continuing from a prior champion is a manual file edit. This feature automates that into a declarative, resumable, budget-capped campaign.

## User Story

As an optimization researcher (Tim)
I want to declare an archipelago campaign in YAML and have an executor chain and merge optimizer runs automatically
So that I can explore many seed lineages in parallel directions and converge them into one best search function, without hand-managing dozens of runs.

## Problem Statement

The optimizer evolves hard *within* a single run (pool, Pareto frontier, COMBINE generations) but has zero *cross-run* navigation. Multiplying runs by hand (re-porting a champion into the base file, choosing which to combine) is slow, error-prone, and unrecorded. There is no way to: (a) auto-continue a branch from its own champion, (b) merge two branches' champions, or (c) see the resulting seed graph.

## Solution Statement

Add **two small engine capabilities** + **one orchestrator**:

1. **Seed-from-champion** — seed a run from a saved champion via `FileSet.with_overlay` (the proven `final_test` pattern), driven by a new config field `seed_champion_path`.
2. **Merge / pool warm-start** — start a run with ≥2 champions already in the candidate pool and force COMBINE mode, driven by new config fields `merge` + `merge_seed_paths`.
3. **Archipelago orchestrator** — a new `cli.py` subcommand `archipelago` that reads a YAML spec, drives runs **sequentially** (no FalkorDB/cache contention), applies the chain-until-converge + tournament-merge rules using `select_holdout.json` as the convergence signal, enforces a global USD budget, is **resumable**, and emits a DAG artifact. A `--dry-run` plans the bracket and writes the DAG **without launching any run** (cheap to validate).

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High (engine touches are small/surgical; the orchestrator + its dynamic DAG logic + tests are the bulk)
**Primary Systems Affected**: `graphretr_opt.config`, `graphretr_opt.campaign`, `graphretr_opt.optimizer.fast_loop`, `graphretr_opt.cli`, new `graphretr_opt.archipelago`
**Dependencies**: None new required (stdlib `subprocess`, `json`, `yaml` already used). Optional: Graphviz `dot` CLI to *render* the emitted `.dot` (the `.dot`/`.json` are written regardless; rendering is optional).

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: YOU MUST READ THESE BEFORE IMPLEMENTING!

- `src/graphretr_opt/cli.py` (lines 25-33 optimize subparser; 82-87 overrides dict; 100,104-105 dispatch) — Why: pattern for adding a subcommand + flags; where overrides are built and `campaign.optimize(...)` is called.
- `src/graphretr_opt/config.py` (lines 20-177 frozen `Config` dataclass; 137-140 `stark_editable_files`; 193-197 `stark_src_abs`; 267-320 `load_config` precedence; 273-276 `fields(Config)` unknown-key validation; 323-331 `dump_resolved_config`) — Why: new fields MUST be declared on the frozen dataclass; precedence = defaults < yaml < env < overrides (overrides win, config.py:307).
- `src/graphretr_opt/campaign.py` (lines 86-145 `_boot_function`, seed built at 129-130; 162-167 `_seed_program`; 175-197 `optimize`; 181-183 run_dir + `dump_resolved_config`; 492-510 `final_test` champion-reload pattern, esp. 506-507 `with_overlay`) — Why: the seed-overlay seam and the canonical champion round-trip.
- `src/graphretr_opt/artifact/file_set.py` (42-52 `from_base`; 54-65 `.sha`; 67-74 `.src`; 77-83 `.primary_file`; 85-87 `.src_of`; 89-92 `with_overlay`; 148-163 `save`; 173-181 `to_dict`/`from_dict`) — Why: `with_overlay` is the insertion point for both seeding and merging; `save` writes ONLY the primary file for a single-file overlay.
- `src/graphretr_opt/optimizer/fast_loop.py` (471 `run` signature; 500 `pool_max_tokens`; 511 pool construct; 580-594 non-resume seed block; 586-590 seed scoring; 591 seed `pool.consider`; 620 `combine_mode = generation > 0`; 689-698 parent/mate selection; 748 `combine_with=mate_prog`; 1004-1025 generation bump/restart; 1062-1071 champion `save`; 275-354 `_final_select`; 356-374 `_cost_aware_pick`; 376-403 `_write_select_audit`) — Why: warm-start injection point (after 591), force-combine point (620), and the bake-off/convergence writer.
- `src/graphretr_opt/optimizer/pool.py` (66 `__init__`; 83-121 `consider` with crash-reject:96, token-wall:101, dedup:104, dominance:106-113, evict:115-117; 142 `select_parent`; 173 `select_mate`; 39 `_sole_best_counts`; 25 `_INSTANCE_KEY="mrr"`) — Why: admission gotchas that can silently drop a warm-start champion.
- `src/graphretr_opt/reward/objectives.py` (34 `MetricVector`; per_query/`.crashed`/`.code_tokens`) — Why: `pool.consider` needs a `MetricVector` with `per_query` populated on the SAME `gate_idxs`.
- `tests/test_final_select.py` (1-30 `_check` + header; 31-46 `_prog`/`_mv` builders; 48-61 `_FakeReward.score`; 62+ `_FakeSubstrate`) — Why: THE template for infra-free unit tests (fakes, no FalkorDB/network) — mirror it for the new tests.
- `tests/test_zai_backend.py` (22-24 `_check`; 12 run recipe in docstring) — Why: dual-mode (pytest + `__main__`) test convention.
- `run3_launch.sh` (uses `setsid nohup ... python -u ... > runs/<name>.log 2>&1 &`, writes `.pid`) — Why: the house headless-launch pattern the orchestrator's subprocess calls should mirror (esp. `python -u`).
- `runs/run13_glm_smoke/select_holdout.json` and `runs/run13_glm_smoke/lineage.jsonl` — Why: real examples of the convergence signal schema to test against.

### New Files to Create

- `src/graphretr_opt/archipelago.py` — the orchestrator: YAML spec loader, sequential run driver (subprocess), chain-until-converge + tournament-merge logic, budget guard, resume, DAG (`graph.json`/`graph.dot`) emitter. House style: `print("[archipelago] ...", flush=True)`.
- `configs/archipelago.yaml` — declarative campaign spec (schema below). The user-facing interface.
- `tests/test_seed_from_champion.py` — unit test: `_boot_function` overlays a champion → seed `.sha` changes and `primary_file` content == champion.
- `tests/test_merge_warmstart.py` — unit test (fakes, no infra): `FastLoop.run` with a 2-program `seed_pool` admits both to the pool and sets `combine_mode=True` so `select_mate` returns the partner.
- `tests/test_archipelago.py` — unit tests for the orchestrator's pure logic: beat-seed convergence recipe, tournament-bracket pairing, DAG artifact shape, `--dry-run` plan — all driven by fake `select_holdout.json`/`lineage.jsonl` in a tmp dir (no real runs).

### Relevant Documentation — READ BEFORE IMPLEMENTING

- Island-model / migration GA background (concept only; no library): https://en.wikipedia.org/wiki/Parallel_metaheuristic#Island_model — Why: confirms the chain-then-merge bracket is the standard archipelago pattern; informs the merge-direction and convergence design.
- Graphviz DOT language: https://graphviz.org/doc/info/lang.html — Why: the `graph.dot` emitter must produce valid DOT; rendering via `dot -Tsvg graph.dot -o graph.svg` is optional.
- (No external Python deps. `yaml` is already used in `config.py`; do not add Airflow/Dagster/Prefect — see NOTES for the rationale.)

### Patterns to Follow

**Naming Conventions:** `snake_case` functions/vars; module-level constants `UPPER_SNAKE`; config fields `snake_case` on the frozen `Config` dataclass. CLI subcommands lowercase (`optimize`, `optimize-search`, `stage0`, `final`, `ablate`, `viz`) → add `archipelago`.

**Print/logging:** `print()` only, NO logging library (zero `logging` imports in `src/`). Bracketed prefixes: `[campaign] ...`, `[fast_loop] ...`. New code uses `print("[archipelago] ...", flush=True)` (flush because output is redirected under nohup; most existing prints are NOT flushed — see fast_loop.py:646 for the one that is).

**Config override flow (cli.py:82-87):** build an `overrides` dict only for keys that map to declared `Config` fields; pass run-specific positionals (`steps`, `campaign`) directly to `campaign.optimize(...)`. Overrides win last (config.py:307).

**Champion round-trip (campaign.py:506-507):**
```python
seed = self._seed_program()
best = seed.with_overlay({seed.primary_file: open(best_path, encoding="utf-8").read()})
```

**Bake-off / artifact-as-truth:** decisions key off JSON artifacts (`select_holdout.json`, `lineage.jsonl`), NOT parsed log text (stdout is block-buffered under nohup unless `python -u`).

**Anti-patterns to avoid:** do NOT mutate the immutable base checkout `starksearch/src` to seed a run (breaks `.sha`/`materialize`); use `with_overlay`. Do NOT abuse `generation>0` to force combine (it's checkpointed and trips `max_generations` stop) — use a dedicated `merge` flag. Do NOT pass FileSets/lists as `overrides` into `Config` unless they're simple serializable types (a path string or tuple-of-path-strings is fine and auto-dumps to `resolved_config.yaml`).

---

## IMPLEMENTATION PLAN

### Phase 1: Engine foundation (seed-from-champion + merge primitives)

Small, surgical changes to the existing engine so the orchestrator has the levers it needs.

**Tasks:**
- Declare new `Config` fields: `seed_champion_path: str = ""`, `merge: bool = False`, `merge_seed_paths: tuple = ()`.
- Overlay a champion onto the seed in `_boot_function`.
- Warm-start the pool from `merge_seed_paths` and force COMBINE in `fast_loop.run`.
- Wire CLI flags `--seed-from-run` / `--seed-champion` / `--merge` / `--merge-seed`.

### Phase 2: Orchestrator core

**Tasks:**
- `archipelago.py`: YAML spec loader + validation; the sequential run driver (subprocess, `python -u`); the convergence recipe; the branch state machine.
- DAG ledger (`graph.json`) + Graphviz (`graph.dot`) emitter; resume from the ledger.
- Global budget guard (reads `runs/openai_usage.json`).

### Phase 3: Integration

**Tasks:**
- New `archipelago` subcommand in `cli.py` dispatch.
- `configs/archipelago.yaml` example spec.
- `--dry-run` mode (plan + DAG, no launches).

### Phase 4: Testing & validation

**Tasks:**
- Unit tests for the two engine seams (fakes, no infra).
- Unit tests for orchestrator logic (fake artifacts in tmp dir).
- `--dry-run` end-to-end smoke (no optimizer launched).

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable. **Do not launch any real optimizer run while `run14_glm` (or any run) is active** — they share one FalkorDB (port 6380) and one `runs/llm_cache.json`.

### UPDATE `src/graphretr_opt/config.py`
- **IMPLEMENT**: Add three fields to the frozen `Config` dataclass (near the strategy/seed fields, ~config.py:129-140): `seed_champion_path: str = ""`, `merge: bool = False`, `merge_seed_paths: tuple = ()`. Add a yaml→tuple coercion for `merge_seed_paths` mirroring `stark_editable_files` (config.py:310-311).
- **PATTERN**: existing tuple field + coercion at config.py:138-140 and 310-311.
- **GOTCHA**: undeclared keys fail either the yaml allowlist (config.py:276) or `Config(**kw)` (config.py:320). Fields auto-dump to `resolved_config.yaml` via `resolved_dict` (config.py:218-221) — good for audit; `merge_seed_paths` as a tuple-of-strings serializes cleanly.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=$PWD/src .venv/bin/python -c "from graphretr_opt.config import load_config; c=load_config(merge=True, seed_champion_path='x', merge_seed_paths=('a','b')); print(c.merge, c.seed_champion_path, c.merge_seed_paths)"`

### UPDATE `src/graphretr_opt/campaign.py` (`_boot_function`, ~line 129)
- **IMPLEMENT**: Immediately after `self.seed = FileSet.from_base(...)` (campaign.py:129-130), add:
  ```python
  if getattr(cfg, "seed_champion_path", ""):
      champ = open(cfg.seed_champion_path, encoding="utf-8").read()
      self.seed = self.seed.with_overlay({self.seed.primary_file: champ})
      print(f"[campaign] seed overlaid from champion {cfg.seed_champion_path} "
            f"-> seed sha={self.seed.sha[:8]}", flush=True)
  ```
- **PATTERN**: identical to `final_test` reload at campaign.py:506-507.
- **IMPORTS**: none new (`FileSet` already imported in boot).
- **GOTCHA**: single-file STaRK overlay only persists the primary file in `best_search.py` (file_set.py:148-163). If `stark_editable_files` ever grows >1, read the `<name>.files/` tree or checkpoint dict instead. The token-cap default (campaign.py:137) is computed from the seed — after overlay the seed IS the champion, so the cap auto-adjusts; good.
- **VALIDATE**: covered by `tests/test_seed_from_champion.py` (below).

### UPDATE `src/graphretr_opt/campaign.py` (`optimize`, ~line 175-194)
- **IMPLEMENT**: Build warm-start FileSets from `cfg.merge_seed_paths` and pass to the loop. After `seed = self._seed_program()` (campaign.py:179-180), construct:
  ```python
  seed_pool = []
  for p in getattr(cfg, "merge_seed_paths", ()):
      src = open(p, encoding="utf-8").read()
      seed_pool.append(self.seed.with_overlay({self.seed.primary_file: src}))
  ```
  Then change the loop call (campaign.py:194) `result = loop.run(self.substrate, seed, steps, campaign)` → `... loop.run(self.substrate, seed, steps, campaign, seed_pool=seed_pool)`.
- **PATTERN**: `with_overlay` (file_set.py:89-92); positional/kw threading like existing `optimize`.
- **GOTCHA**: keep `seed_pool` as `[]` default so non-merge runs are unaffected.
- **VALIDATE**: `tests/test_merge_warmstart.py`.

### UPDATE `src/graphretr_opt/optimizer/fast_loop.py` (`run` signature ~471, warm-start ~after 591, combine ~620)
- **IMPLEMENT (a)**: add param to `run`: `def run(self, substrate, seed_program, steps, campaign, seed_pool=None) -> ArmResult:` (fast_loop.py:471).
- **IMPLEMENT (b)**: force combine — change fast_loop.py:620 to `combine_mode = generation > 0 or bool(getattr(self._cfg, "merge", False))` (use whatever the in-scope cfg handle is named in `run`; confirm — it's `cfg`).
- **IMPLEMENT (c)**: warm-start injection — inside the `if not resumed:` block, immediately after `pool.consider(best_prog, best)` (fast_loop.py:591), add:
  ```python
  for wp in (seed_pool or []):
      if wp.sha in pool.shas():
          continue
      wfn = self._sandbox.compile(wp.src)
      self._sandbox.probe(wfn, probes, cfg.probe_timeout_s)
      fn_cache[wp.sha] = wfn
      wmv = cache.get(_ckey(wp.sha))
      if wmv is None:
          wmv = self._reward.score(wfn, gate_idxs, src=wp.src,
                                   per_query_timeout_s=cfg.probe_timeout_s)
          cache.put(_ckey(wp.sha), wmv)
      admitted = pool.consider(wp, wmv)[1] if not getattr(wmv, "crashed", False) else False
      print(f"[fast_loop] merge warm-start {wp.sha[:8]} "
            f"recall@20={wmv.get('recall@20'):.4f} admitted={admitted}", flush=True)
  ```
- **PATTERN**: seed scoring at fast_loop.py:586-590; admission at 591; `_ckey`/`cache` already in scope.
- **GOTCHA (token wall)**: `pool.consider` rejects `code_tokens > pool.max_tokens` (pool.py:101; wall = `cfg.gate_max_tokens`, fast_loop.py:500). A champion from a different architecture may exceed it. MITIGATION: before the warm-start loop, recompute the wall from the largest champion, e.g. set the pool's `max_tokens` to `max(pool.max_tokens, 1.3 * max(code_tokens(wp.src_of(wp.primary_file)) for wp in seed_pool))` (import `code_tokens` from `..reward.objectives` as campaign.py:136 does), OR temporarily set `pool.max_tokens = 0` for the warm-start then restore. Document the choice in a comment.
- **GOTCHA (dominance)**: if one champion dominates the other and the other is sole-best on 0 queries, only one survives → `select_mate` returns None → no crossover. After warm-start, if `cfg.merge` and `len(pool) < 2`, `print("[fast_loop] merge: <2 distinct champions admitted; combine inert", flush=True)` (don't crash; the run degrades to a normal single-seed run).
- **GOTCHA (epoch)**: score warm members on the SAME `gate_idxs` as the seed so `per_query` keys align for sole-best counting (pool.py:49-61).
- **VALIDATE**: `tests/test_merge_warmstart.py`.

### UPDATE `src/graphretr_opt/cli.py` (optimize subparser ~25-33, overrides ~82-87)
- **IMPLEMENT**: add flags to the `optimize` subparser: `--seed-from-run` (str, default None — resolves to `runs/<name>/best_search.py`), `--seed-champion` (str path, default None), `--merge` (`action="store_true"`), `--merge-seed` (`nargs="+"`, default None — explicit champion paths OR run names). In the overrides block (cli.py:82-87): set `overrides["seed_champion_path"]` from `--seed-champion` or the resolved `--seed-from-run` path; set `overrides["merge"]=True` and `overrides["merge_seed_paths"]=tuple(resolved_paths)` from `--merge`/`--merge-seed` (resolve bare run names to `runs/<name>/best_search.py`).
- **PATTERN**: cli.py:25-33 (args) and 82-87 (overrides).
- **GOTCHA**: resolve run-name → `runs/<name>/best_search.py` using `load_config().runs_dir` (or the campaign helper) so relative paths are correct; error clearly if the champion file is missing (mirror campaign.py:501 `SystemExit`).
- **VALIDATE**: `PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli optimize --help` shows the new flags.

### CREATE `src/graphretr_opt/archipelago.py`
- **IMPLEMENT**: the orchestrator. Key functions:
  - `load_spec(path) -> dict` — read YAML, apply defaults (`branches: 5`, etc.), validate.
  - `run_one(campaign_name, steps, *, seed_from=None, merge_seeds=None, checkpoint_every, dry_run) -> dict` — `subprocess.run([... python -u -m graphretr_opt.cli optimize --steps S --campaign-name NAME (--seed-from-run R | --seed-champion P) (--merge --merge-seed ...) --checkpoint-every K], cwd=graphretr_demo, env=PYTHONPATH)`, blocking; on return read `runs/<NAME>/select_holdout.json` + `lineage.jsonl`. In `dry_run`, skip subprocess and return a planned node.
  - `beat_seed(run_dir, margin) -> (bool, champ_h, seed_h)` — the convergence recipe (below).
  - `run_branch(branch_id, seed0, spec) -> champion_run` — chain: run, beat-seed test, chain from `best_search.py` until not-beaten or `max_runs_per_branch`.
  - `merge_bracket(champions, spec) -> champion` — pairwise: for each pair, a merge run warm-started with both `best_search.py` files; recurse on winners; odd one out byes to next round.
  - `budget_ok(spec) -> bool` — read `runs/openai_usage.json` `usd` vs `spec.execution.budget_usd`; stop scheduling when exceeded.
  - DAG: append nodes/edges to `runs/<campaign>/archipelago_graph.json` after each run; re-emit `graph.dot`. On start, load the ledger and SKIP runs that already have `select_holdout.json` (resume).
- **CONVERGENCE RECIPE (copy-ready)**:
  ```python
  seed_sha = json.loads(open(run_dir+"/lineage.jsonl").readline())["parent_sha"]
  audit = json.load(open(run_dir+"/select_holdout.json"))
  by_sha = {m["sha"]: m for m in audit["members"] if not m["crashed"]}
  seed_h = by_sha.get(seed_sha, {}).get("holdout_value")
  champ_h = by_sha[audit["exported_sha"]]["holdout_value"]
  beat = (seed_h is not None) and (champ_h > seed_h + margin) and (audit["exported_sha"] != seed_sha)
  ```
- **PATTERN**: house style `print("[archipelago] ...", flush=True)`; artifacts-as-truth; `subprocess` with `python -u` (run3_launch.sh).
- **IMPORTS**: `json, os, subprocess, sys, yaml` (+ `from .config import load_config` for `runs_dir`).
- **GOTCHA**: sequential only (`execution.mode: sequential`) — never launch two runs concurrently (FalkorDB + `llm_cache.json` are single, shared). Use blocking `subprocess.run`, check `returncode`, and treat a missing/`crashed` `select_holdout.json` as a failed run (don't chain off it).
- **VALIDATE**: `tests/test_archipelago.py` + `--dry-run` smoke.

### CREATE `configs/archipelago.yaml`
- **IMPLEMENT**: the declarative spec (schema in NOTES). Include comments so it's self-documenting.
- **VALIDATE**: `PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli archipelago --spec configs/archipelago.yaml --dry-run` prints the planned DAG and writes `graph.dot`/`graph.json`, launching nothing.

### UPDATE `src/graphretr_opt/cli.py` (dispatch ~99-105)
- **IMPLEMENT**: add subparser `archipelago` with `--spec` (required), `--dry-run` (`store_true`), `--resume` (`store_true`). In dispatch, `from .archipelago import run_campaign; run_campaign(args.spec, dry_run=args.dry_run, resume=args.resume)`.
- **PATTERN**: existing subparser/dispatch (cli.py:25-46, 76-105).
- **VALIDATE**: `... cli.py archipelago --help`.

### CREATE `tests/test_seed_from_champion.py`
- **IMPLEMENT**: build a base `FileSet.from_base` over `cfg.stark_src_abs`, capture `seed.sha`; write a tiny modified champion to a tmp file; call the overlay path (either via `Campaign` with `seed_champion_path` set, or directly exercise `with_overlay`); assert new `.sha != base.sha` and `seed.src_of(primary) == champion_src`.
- **PATTERN**: `tests/test_file_set.py` + `tests/test_stark_fileset_seed.py`; `_check` harness.
- **VALIDATE**: `.venv/bin/pytest tests/test_seed_from_champion.py`

### CREATE `tests/test_merge_warmstart.py`
- **IMPLEMENT**: mirror `tests/test_final_select.py` fakes (`_FakeReward`, `_FakeSubstrate`, `_prog`, `_mv` with populated `per_query` on the gate idxs). Construct a `FastLoop` with a `cfg` where `merge=True`, run `run(..., seed_pool=[progA, progB])` for `steps=0` (or a tiny stubbed loop), and assert: both champions admitted (`len(pool) == >=2` including seed), `combine_mode` True, and `pool.select_mate(rng, exclude_sha=A.sha)` returns B. Also assert the token-wall mitigation admits an oversized champion.
- **PATTERN**: `tests/test_final_select.py:48-61` fakes; `tests/test_pool_select.py` for pool assertions.
- **GOTCHA**: give the two fake `_mv`s distinct per-query mrr winners so neither dominates the other (so both survive `consider`).
- **VALIDATE**: `.venv/bin/pytest tests/test_merge_warmstart.py`

### CREATE `tests/test_archipelago.py`
- **IMPLEMENT**: pure-logic tests with fake artifacts written to a tmp `runs/` dir (NO optimizer launched): (1) `beat_seed` returns False when `exported_sha == seed_sha` (the run13 case) and True when champion's `holdout_value > seed + margin`; (2) `merge_bracket` pairs `[c1,c2,c3,c4]`→`[(c1,c2),(c3,c4)]`→1 winner, with an odd count byeing correctly; (3) the DAG `graph.json`/`graph.dot` contain the expected nodes/edges; (4) `--dry-run` plans N branches and launches nothing (assert `subprocess` not called — monkeypatch/inject a fake runner).
- **PATTERN**: `_check` harness; build fake `select_holdout.json`/`lineage.jsonl` matching the verified schema.
- **VALIDATE**: `.venv/bin/pytest tests/test_archipelago.py`

---

## TESTING STRATEGY

Mirror the project's **dual-mode, infra-free** convention: every test is pytest-discoverable AND runnable as `python -m tests.<name>`, using fakes (no FalkorDB, no network, no LLM). `tests/test_final_select.py` is the gold template.

### Unit Tests
- `test_seed_from_champion.py` — overlay correctness (sha change + content).
- `test_merge_warmstart.py` — pool warm-start admission + forced COMBINE + `select_mate`.
- `test_archipelago.py` — convergence recipe, bracket pairing, DAG artifact, dry-run.

### Integration Tests
- `--dry-run` of the full `archipelago` subcommand against `configs/archipelago.yaml` (plans the DAG, writes `graph.json`/`graph.dot`, launches nothing). This is the only "integration" test runnable without spending money/compute.
- A REAL 1-branch / 1-run smoke (`branches: 1`, `steps: 2`, `max_runs_per_branch: 1`) is a manual acceptance step, **run only after `run14_glm` finishes** (shared FalkorDB/cache), and is NOT part of the automated suite.

### Edge Cases
- Run produces no `select_holdout.json` (crashed/killed) → branch fails gracefully, not chained.
- Seed crash-excluded from bake-off (`seed_sha` absent from `members`) → treat as not-beaten.
- Champion exceeds `gate_max_tokens` → warm-start token-wall mitigation admits it.
- Two champions where one dominates the other → `len(pool) < 2` → combine inert, run degrades to single-seed (logged, no crash).
- Odd number of branch champions in a merge round → bye to next round.
- Global budget exceeded mid-campaign → stop scheduling, finalize DAG, exit cleanly.
- Resume: re-run the campaign → skip runs that already have `select_holdout.json`.

---

## VALIDATION COMMANDS

Run from `cd "/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo"`. Execute every command; expect zero errors.

### Level 1: Syntax & Style
```bash
PYTHONPATH=$PWD/src .venv/bin/python -c "import graphretr_opt.config, graphretr_opt.campaign, graphretr_opt.optimizer.fast_loop, graphretr_opt.archipelago, graphretr_opt.cli; print('import ok')"
PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli optimize --help
PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli archipelago --help
```

### Level 2: Unit Tests
```bash
.venv/bin/pytest tests/test_seed_from_champion.py tests/test_merge_warmstart.py tests/test_archipelago.py -q
```

### Level 3: Regression (whole suite — no infra tests must break)
```bash
.venv/bin/pytest tests/ -q
```

### Level 4: Manual Validation (no spend)
```bash
PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli archipelago --spec configs/archipelago.yaml --dry-run
# expect: a printed plan of N branches + merge bracket, and runs/<campaign>/graph.dot + graph.json written; NO optimizer process launched.
dot -Tsvg runs/<campaign>/graph.dot -o /tmp/graph.svg   # optional render (if graphviz installed)
```

### Level 5: Manual Acceptance (spends money/compute — ONLY after run14 finishes, with user approval)
```bash
# minimal real run: 1 branch, 2 steps, no merge
PYTHONPATH=$PWD/src .venv/bin/python -m graphretr_opt.cli archipelago --spec configs/archipelago_smoke.yaml
# verify the chained run's select_holdout.json + the DAG node/edge were recorded.
```

---

## ACCEPTANCE CRITERIA

- [ ] `--seed-from-run`/`--seed-champion` seeds a run from a champion (seed sha == champion's overlay sha in `resolved_config`/log).
- [ ] `--merge --merge-seed a b` warm-starts both champions into the pool and runs in COMBINE mode from step 0.
- [ ] `archipelago --dry-run` plans N (default 5, configurable) branches + the merge bracket and writes `graph.json`/`graph.dot`, launching nothing.
- [ ] Real campaign runs **sequentially**, chains each branch until not-beaten (correct beat-seed recipe), then merges via tournament bracket to one champion.
- [ ] Global `budget_usd` halts scheduling cleanly; campaign is resumable (skips completed runs).
- [ ] All Level 1-4 validation commands pass; full existing suite still green (no regressions).
- [ ] New code matches house style (`print("[archipelago] ...", flush=True)`, snake_case, no logging lib).

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task's VALIDATE passed immediately
- [ ] Level 1-4 validation commands succeed
- [ ] Unit + dry-run integration tests pass; existing suite unbroken
- [ ] No lint/type errors (project has no enforced type-checker; keep imports clean)
- [ ] Manual `--dry-run` confirms DAG + plan
- [ ] Acceptance criteria all met
- [ ] Reviewed for quality/maintainability

---

## NOTES

**Why a YAML-spec + thin sequential subcommand, not Airflow/Dagster/Prefect.** The workflow is *dynamic* (chain length and merge bracket depend on runtime bake-off results), which static-DAG tools model poorly; the dynamic-capable engines are heavy infra (daemon + scheduler + DB + web UI) for what is fundamentally a sequential loop shelling out to a long subprocess and reading one JSON. A YAML spec gives the declarative interface; the orchestrator emits a real, visualizable DAG (`graph.dot`) as an artifact. If a named engine is later required, Prefect is the lightest dynamic option, but it's out of scope here.

**Sequential by design.** One FalkorDB (port 6380) and one `runs/llm_cache.json` are shared. The cache-write fix (`stark_harness/qa_runner.py`, threading.Lock + unique temp) is crash-safe *across processes* (atomic `os.replace` of a unique temp) but a per-process lock — concurrent runs would still churn the cache (lost writes, a perf loss, not corruption). So `execution.mode: sequential` is the supported mode; parallel is explicitly deferred (would need per-run cache files or a shared cache server).

**Two flags vs config fields.** `seed_champion_path` (str) and `merge`/`merge_seed_paths` (bool/tuple) are declared `Config` fields so they auto-dump to `resolved_config.yaml` (auditable, affects `config_hash` → honest experiment id). `merge_seed_paths` as a tuple-of-path-strings serializes cleanly; this is the one deviation from the "don't put lists in overrides" caution, justified by auditability.

**`merge_seed_paths` resolution.** The CLI resolves bare run names → `runs/<name>/best_search.py`; the orchestrator passes explicit paths. The engine treats them as file paths and overlays each onto a fresh `from_base` FileSet.

**DAG schema (`graph.json`).** `{"campaign": str, "nodes": [{"id": run_name, "kind": "chain"|"merge", "branch": int, "seed_from": [parent run ids], "champion_sha": str, "holdout_value": float, "beat_seed": bool}], "edges": [[parent_run, child_run]], "champion": run_name}`. `graph.dot` renders nodes labeled with run name + holdout value, edges = seed/merge lineage.

**`configs/archipelago.yaml` schema (example):**
```yaml
branches: 5                 # configurable; any N
seed: base                  # 'base' = current editable file; or per-branch champion paths
strategy_per_branch: []     # optional: push branches different directions (strategy names)
run:
  steps: 20
  checkpoint_every: 5
  backend: zai              # GLM (campaign.yaml already pins this; informational)
converge:
  metric: holdout_value     # the select_holdout.json comparison currency
  margin: 0.01              # champion must beat seed by this to continue the chain
  max_runs_per_branch: 4    # safety cap on chain length
merge:
  policy: tournament        # pairwise (1,2)(3,4)... then winners, recursively
execution:
  mode: sequential          # ONLY supported mode (shared FalkorDB/cache)
  budget_usd: 25            # hard ceiling across the whole campaign (reads runs/openai_usage.json)
```

**Operational gate.** Do NOT run Level 5 (or any real launch) until `run14_glm` completes — shared FalkorDB/cache. Editing source files does not affect the already-running `run14` process (modules were imported at its start), so Phase 1-4 code work is safe to do now; only *launching* runs is gated.

**Confidence: 8/10** for one-pass implementation. The engine seams are precisely located and low-risk; the residual risk is in the orchestrator's dynamic bracket/resume logic and the two warm-start gotchas (token wall, dominance), all of which have explicit mitigations and unit tests above.
