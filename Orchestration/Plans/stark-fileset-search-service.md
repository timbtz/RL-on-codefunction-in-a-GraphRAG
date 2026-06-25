# Feature: Evolve the STaRK retrieval as a whole editable search service (FileSet), like `graphsearch`

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files etc. This plan touches **two sibling packages** under the repo root `Demo 1/`: the optimizer engine `graphretr-demo/` and the STaRK service package `starksearch/`. The reference implementation to MIRROR is the `graphsearch/` package + its `graph_search` engine path.

## Feature Description

Today the STaRK optimizer evolves a single `def search(q, G) -> dict[int,float]` string that calls a frozen, heavyweight `G` (`RetrievalGraph`) DSL: ~15 fixed primitives, each with its own caching/metering/allowlist/query-pinning (`starksearch/graph.py`, 781 lines), plus a `primitives.py` validation wall. In v7 the candidate already reimplements every operator on top of three raw primitives (`G.query`/`G.embed`/`G.llm`), so the entire convenience-operator layer of `G` is dead weight the candidate routes around.

The **company-KG path (`graph_search`) already does it the right way**: the candidate is a real, multi-file **`FileSet`** (a throwaway copy of `graphsearch/src` with the edited service file overlaid), run in an isolated subprocess; `Neo4jGraph` + a chat model are constructor-injected and wrapped in transparent cost-metering proxies (`graphsearch/src/common/service/qa_eval/qa_runner.py`); isolation is the process boundary + a wall-clock kill, with no AST sandbox and no DSL.

This feature makes the STaRK path work **identically to `graph_search`**: the evolved artifact becomes a whole `StarkGraphSearchService` class in a `starksearch/src/` base tree, FalkorDB + embedder + a metered LLM client are injected, the container-protecting caps move into a read-only DB proxy (a firewall), and the `G` DSL + the in-process sandbox/RewardModel are deleted.

## User Story

As an **optimizer operator running the STaRK biomedical benchmark**
I want **the candidate to be the entire editable search service (real files, real imports), not a sandboxed `search(q, G)` calling a frozen DSL**
So that **the optimizer can rewrite the whole retrieval pipeline end-to-end exactly as it already does for the Neo4j company-KG service — with less overhead, no phantom sandbox rules, and the recall ceiling fully in scope.**

## Problem Statement

1. **Dead, duplicated layer.** `starksearch/graph.py:213-633` exposes `vector_search/get_neighbors/k_hop_expand/filter_nodes/shortest_path/rank_by_text/text_search/extract/llm_rerank/reformulate/judge_sufficient/pick_frontier` as frozen methods. The v7 seed (`starksearch/seeds/reasoning_first_v7.py`) reimplements all of them itself on `G.query/embed/llm` and calls none of them. ~500 lines are unused on the live path yet still built in-parent and rebuilt in every worker.
2. **Two scorers + a phantom sandbox.** `reward/evaluator.py::RewardModel` still calls an in-process `self._sandbox.run` (AST gate, `SAFE_BUILTINS`, `SandboxError`); `reward/subprocess_reward.py::StarkRewardAdapter` is a near-duplicate that runs the subprocess. The active path uses `NullSandbox` (pass-through) but the AST machinery still ships.
3. **The candidate is lied to.** The v7 seed docstring (`reasoning_first_v7.py:28-29`) imposes "NO imports, NO underscore attributes, only these builtins, bare `except:`" — AST-gate rules that no longer apply (`_worker_stark.py:57` execs with real builtins/free imports). The optimizer is needlessly constrained.
4. **Single-string artifact, not a service.** `StarkSubprocessTarget.run` passes one `candidate_src` string; the `FileSet` machinery the engine already has (and `graph_search` uses) is bypassed, so STaRK can't evolve multi-file real services.
5. **Latent safety hole.** The `algo.SPpaths maxLen<=3` wall lives only in `RetrievalGraph.shortest_path` — which v7 doesn't call. A mutated candidate can emit `algo.SPpaths(... maxLen:5)` through raw `G.query` and peg the **shared** FalkorDB; the subprocess wall-clock kill does NOT stop a runaway server-side query (per `starksearch/README.md`, "Contract").

## Solution Statement

Re-point the STaRK path onto the engine's existing `FileSet` + `SubprocessSearchTarget` machinery, mirroring `graph_search`:

- The evolved artifact becomes a **`StarkGraphSearchService(db, embedder, llm)`** class in a new `starksearch/src/` base tree (the v7 seed logic ported verbatim into methods). `editable_files` = just that service file.
- A new **`starksearch/src/stark_harness/qa_runner.py`** (mirror of `graphsearch`'s `qa_runner.py`) builds the service from `falkor_cfg`/`llm_cfg`, injecting: a **read-only metered FalkorDB proxy** (`db.cypher(query, params)`), the **embedder** (`embedder.encode(text)`), and a **metered+cached JSON LLM client** (`llm(system, user, ...)`). Metering goes into a `CostSink`; **the SPpaths/write firewall + row/timeout caps live in the proxy** (harness, non-editable), not in a method the candidate can bypass.
- The STaRK worker is rewritten to materialize the FileSet overlay, build the metered service, and return per-query `{pred, cost, error}` (pred = `dict[int,float]`).
- **Delete** the `G` DSL operator layer, `primitives.py`'s operator-arg validation, the in-process `RewardModel`/AST sandbox, and the phantom seed rules.

Keep what is load-bearing: read-only backend, row/timeout caps, the SPpaths wall (moved to the proxy), the LLM budget ceiling + disk cache, the eval-hygiene seam (only question strings cross), and `StarkRewardAdapter` as the **single** scorer.

## Feature Metadata

**Feature Type**: Refactor (architecture migration; behavior-preserving on quality)
**Estimated Complexity**: High
**Primary Systems Affected**: `starksearch/` (new `src/` service tree + harness; delete `graph.py`/`primitives.py` DSL), `graphretr-demo/src/graphretr_opt/` (`env/targets/_worker_stark.py`, `env/targets/stark_subprocess_target.py`, `campaign.py::_boot_function`, `config.py`, delete `reward/evaluator.py` in-process path + `env/null_sandbox.py` AST remnants)
**Dependencies**: `falkordb`, `numpy`, the existing embedder (`starksearch/embedder.py`), `torch` (STaRK evaluator), `mlflow`, the mutator LLM backend. No new external deps.

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: YOU MUST READ THESE BEFORE IMPLEMENTING!

**The pattern to mirror (graph_search / graphsearch):**
- `graphretr-demo/src/graphretr_opt/artifact/file_set.py` (whole file) — Why: the `FileSet` artifact (overlay on an immutable base, `from_base`, `materialize`, `sha`, `with_overlay`). STaRK will seed `FileSet.from_base(stark_src_abs, editable_files)`; do not modify this class.
- `graphretr-demo/src/graphretr_opt/env/targets/subprocess_target.py` (whole file) — Why: the production target that materializes a FileSet overlay, spawns the worker, enforces the wall-clock kill, removes the throwaway dir. The STaRK target must adopt this `run(file_set, ...)` shape (currently it takes a `src` string).
- `graphretr-demo/src/graphretr_opt/env/targets/_worker.py` (whole file) — Why: the worker shape to mirror — `sys.path.insert(0, overlay_dir)` first, import the harness from the overlay, build `(service, sink)` slots, `_one()` per query, optional `query_concurrency`. STaRK worker mirrors this but returns `pred` dicts.
- `graphsearch/src/common/service/qa_eval/qa_runner.py` (whole file) — Why: THE template for the STaRK harness — `CostSink`, `_MeteredGraph`/`_MeteredChat` proxies, `build_service(cfg, instrument=...)`, `run_query(service, q)`. The STaRK `qa_runner` is the FalkorDB/embedder/LLM analogue.
- `graphretr-demo/src/graphretr_opt/campaign.py:161-233` (`boot_search`) and `:235-278` (`_search_cfg`) — Why: how `graph_search` wires substrate + target + reward + `FileSet.from_base` + gate-knob re-derivation. STaRK's `_boot_function` (`:61-112`) will be rewritten to mirror this.
- `graphretr-demo/src/graphretr_opt/optimizer/mutator.py:38-42, 173-272` — Why: the mutator is ALREADY artifact-agnostic — it duck-types a FileSet via `hasattr(program, "overlay")` and renders/edits multiple files. **No mutator change is needed**; just hand it a FileSet + a domain note (like `SEARCH_DOMAIN_NOTE`).

**The STaRK code being migrated/deleted:**
- `starksearch/seeds/reasoning_first_v7.py` (whole file) — Why: the seed logic to PORT into the new service class methods (extract / vector_search / text_search / llm_rerank / reformulate / graph_recall / get_neighbors / filter_nodes / `search`). Behavior must be preserved (parity test).
- `starksearch/graph.py:136-209` (the raw floor: `query`, `embed`, `llm`) and `:30, 110-122` (row cap, engine config, allowlist bootstrap) — Why: the only parts that survive, refactored into the harness `db`/`embedder`/`llm` clients + the firewall. `:213-781` (operator layer + `describe`) is DELETED.
- `starksearch/primitives.py` (whole file) — Why: `nonempty_str`/`as_int`/`clamp`/`ids_in` move into the service (its own input hygiene); `Allowlists` becomes data the service reads; the **caps/firewall** move into the DB proxy. Then delete.
- `starksearch/backends/falkordb.py` + `backends/base.py` (whole files) — Why: the read-only backend stays; the firewall wraps it. Note `ro_query` already gives read-only + `query_count` metering.
- `graphretr-demo/src/graphretr_opt/env/targets/_worker_stark.py` (whole file) — Why: rewritten to use FileSet overlay + the new harness (currently builds `RetrievalGraph` + execs a single `src` string; `_one` at `:93-116` shows the per-query cost-delta snapshot to preserve).
- `graphretr-demo/src/graphretr_opt/env/targets/stark_subprocess_target.py` (whole file) — Why: `run(src,...)` → `run(file_set,...)` with `file_set.materialize(overlay_dir)`; keep its pred/cost parsing (`:95-109`).
- `graphretr-demo/src/graphretr_opt/reward/subprocess_reward.py` (whole file) — Why: `StarkRewardAdapter` is the KEPT scorer. Its `score(fn, idxs, src=..., ...)` signature stays; it must accept the FileSet as `src` (it already ignores `fn` and forwards `src` to `target.run`). Confirm `target.run` is called with the FileSet, not a string.
- `graphretr-demo/src/graphretr_opt/reward/evaluator.py` (whole file) — Why: `RewardModel` (in-process `self._sandbox.run`) is DELETED. `QUALITY_KEYS` (`:20`) is imported by `subprocess_reward.py` — move it (see Task 8) before deleting.
- `graphretr-demo/src/graphretr_opt/campaign.py:61-112` (`_boot_function`) and `:114-131` (`_make_mutator`, `_seed_program`, `_probe_queries`) — Why: the STaRK bring-up to rewrite. Note `_make_mutator` (`:114-123`) currently passes `self.graph.describe()`; replace with a static `STARK_DOMAIN_NOTE`. `_seed_program` (`:125-127`) uses `load_strategy`+`SearchProgram.from_file`; replace with `FileSet.from_base`.
- `graphretr-demo/src/graphretr_opt/config.py:103-112, 143-169` — Why: add `stark_src`/`stark_editable_files` + `*_abs` properties mirroring `graphsearch_src`/`graphsearch_src_abs`/`editable_files`.
- `graphretr-demo/src/graphretr_opt/env/null_sandbox.py` (whole file) — Why: `NullSandbox`/`NullGraph` are pass-through shims for FastLoop's `compile/probe` + `graph.describe/get_text`. KEEP `NullSandbox` (the FileSet path needs the no-op compile/probe — `graph_search` uses it at `campaign.py:226`); the in-process `RetrievalGraph` it replaces is gone.
- `graphretr-demo/src/graphretr_opt/env/errors.py` — Why: `SandboxError`/`SAFE_BUILTINS`. After deleting `RewardModel`, check remaining importers (`grep -rn SAFE_BUILTINS src`); `_make_mutator` passes `SAFE_BUILTINS` to `Mutator` — on the FileSet path pass `safe_builtins={}` like `graph_search` (`campaign.py:325`).

**Tests to mirror / update:**
- `graphretr-demo/tests/test_stark_parity.py` (whole file) — Why: the regression crux. It drives `StarkRewardAdapter -> StarkSubprocessTarget -> _worker_stark` for the v7 seed against `tests/golden/stark_parity_v7.json`. After migration it must pass with the FileSet seed (re-record golden only if ranking provably unchanged). `SEED_PATH` (`:30`) becomes the service file path.
- `graphretr-demo/tests/test_file_set.py` — Why: FileSet unit-test patterns to reuse for the STaRK seed.
- `graphretr-demo/tests/test_subprocess_target_smoke.py` — Why: offline worker-smoke pattern (no infra) to mirror for the STaRK worker.
- `graphretr-demo/tests/test_agentic_primitives.py` — Why: existing primitive/cap tests; the firewall test (SPpaths/writes rejected) extends this style.

### New Files to Create

- `starksearch/src/stark_search/stark_graph_search_service.py` — the editable `StarkGraphSearchService` class (ported v7 logic). **The single editable artifact.**
- `starksearch/src/stark_search/__init__.py` — package marker.
- `starksearch/src/stark_harness/qa_runner.py` — `build_service`, `run_query`, `CostSink`, `ReadOnlyGraphClient` (metered + firewall proxy over `FalkorDBBackend`), `LlmClient` (metered+cached JSON LLM, budget ceiling), `Embedder` adapter. Non-editable.
- `starksearch/src/stark_harness/__init__.py` — package marker.
- `graphretr-demo/tests/test_stark_firewall.py` — unit tests: writes rejected, `SPpaths maxLen>=4` rejected/capped, row cap enforced, read-only passes.
- `graphretr-demo/tests/test_stark_fileset_seed.py` — the new service file imports, exposes `StarkGraphSearchService`, and `build_service` constructs it offline (monkeypatched backend/llm).

### Relevant Documentation — READ BEFORE IMPLEMENTING!

- `starksearch/README.md` (the "Contract" + "Provenance" sections) — Why: documents the exact caps that protect the SHARED FalkorDB (read-only, row caps, `SPpaths maxLen<=3` wall) and states the wall-clock kill does NOT stop a server-side query. This is the rationale for the firewall-in-proxy.
- `graphsearch/README.md` ("Construction (no DI container)") — Why: shows the inject-constructed-objects pattern (`graph=..., chat_model=...`) the STaRK service must follow.
- FalkorDB engine quirks in `starksearch/backends/falkordb.py:1-9` — Why: vector `score` is cosine DISTANCE (convert `1.0 - score`); `SPpaths maxLen>=4` runs away and ignores the query timeout. The firewall + the ported `vector_search` depend on both.

### Patterns to Follow

**Inject-and-meter (the core pattern), from `graphsearch/src/common/service/qa_eval/qa_runner.py:72-110`:**
```python
class _MeteredGraph:               # transparent proxy: counts .query, delegates the rest
    def __init__(self, graph, sink): self._graph, self._sink = graph, sink
    def query(self, *a, **k): self._sink.db_queries += 1; return self._graph.query(*a, **k)
    def __getattr__(self, name): return getattr(self._graph, name)
```
For STaRK the proxy ALSO firewalls (the new bit): reject writes, cap rows, reject `SPpaths maxLen>=4`.

**FileSet seed + target wiring, from `campaign.py:207-226`:**
```python
target = SubprocessSearchTarget(neo4j_cfg=..., llm_cfg=..., opt_src_dir=cfg.opt_src_abs,
    service_relpath=cfg.editable_files[0], query_concurrency=cfg.query_concurrency,
    worker_module="...._worker_stark")           # SAME target class, STaRK worker
self.search_seed = FileSet.from_base(cfg.stark_src_abs, cfg.stark_editable_files)
```

**Worker per-query metering, from `_worker.py:32-43` (reset sink → run → snapshot) — preserves the per-query cost isolation that `_worker_stark.py:93-116` did via getattr deltas.**

**Naming conventions:** snake_case modules/functions, PascalCase classes; relpaths in `editable_files` are POSIX-style relative to the base `src/` (`file_set.py:39` normalizes). Service method contract: `search(self, query: str) -> dict[int, float]` (node id → score), mirroring `graphsearch`'s `search(query) -> str`.

**Error handling:** a per-query exception in the worker is caught and serialized as `{"error": ...}` → scored as a miss (`_worker.py:39-42`, `subprocess_reward.py:66-74`); never fatal. A build/import failure is the worker's top-level `{"error": ..., "trace": ...}` (`_worker.py:96-100`).

**Eval hygiene:** only question strings cross the seam (`search_target.py:76-78`). Do not pass gold/answer ids to the worker.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — the `starksearch/src/` base tree + harness

Create the importable base tree the FileSet overlays, and the non-editable harness that injects + meters + firewalls FalkorDB/embedder/LLM. No engine wiring yet; everything is unit-testable offline with a fake backend/LLM.

### Phase 2: Core Implementation — port the v7 seed into the service class

Move every v7 helper into `StarkGraphSearchService` methods calling `self._db.cypher` / `self._emb.encode` / `self._llm`. Drop the phantom sandbox rules. Preserve the exact scoring/levers so ranking is unchanged.

### Phase 3: Integration — re-point the engine STaRK path to FileSet

Rewrite the worker + target to materialize/run a FileSet, rewrite `_boot_function` to seed a FileSet and inject the harness cfg, add config fields + gate-knob re-derivation, and make `StarkRewardAdapter` carry the FileSet through as `src`.

### Phase 4: Cleanup & Testing — delete the DSL/sandbox, add firewall/parity guards

Delete `graph.py` operator layer, `primitives.py`, `RewardModel`/in-process sandbox path, phantom rules; add firewall + fileset-seed tests; re-green the parity test.

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable. Run with `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest ...` (matches `pyproject.toml:15` `pythonpath = ["src", ".."]` and `test_stark_parity.py:21`).

### CREATE `starksearch/src/stark_harness/qa_runner.py`
- **IMPLEMENT**: `CostSink` (dataclass: `llm_calls, db_queries, rerank_items, tokens_in, tokens_out, usd_cost`; `reset()`, `snapshot()`) — mirror `graphsearch qa_runner.py:40-57` plus `rerank_items` (the STaRK deterministic cost meter, from `graph.py:91-95`). `ReadOnlyGraphClient(backend, sink, cfg)` exposing `cypher(query, params=None) -> list[list]` that: (1) runs the FIREWALL (`_assert_safe(query)`), (2) `sink.db_queries += 1`, (3) returns `backend.ro_query(query, params or {}, timeout_ms=cfg.query_timeout_ms)` truncated to a row cap (`_MAX_QUERY_ROWS = 5000`, from `graph.py:30`). Expose `embed(text)` delegating to the embedder, and read-only `labels/rel_types/ntypes` allowlist properties (bootstrapped exactly as `graph.py:114-122`). `LlmClient(budget, cfg, sink, runs_dir)` with `__call__(system, user, context_ids=None, model=None, max_tokens=600) -> dict` — port `graph.py:166-209` (metered, disk-cached by `(model,system,user)`, budget ceiling) and increment `sink.llm_calls`/`sink.rerank_items`. `build_service(falkor_cfg, cfg_overrides, instrument=None) -> StarkGraphSearchService`: load_config, build `FalkorDBBackend`, embedder, budget, wrap in clients (metered iff `instrument`), return the service. `run_query(service, query) -> dict[int,float]` calling `service.search(query)`.
- **PATTERN**: `graphsearch/src/common/service/qa_eval/qa_runner.py:40-110, 222-257` (CostSink/proxies/build_service/run_query); `starksearch/graph.py:114-122, 144-209` (allowlist bootstrap + raw floor to port).
- **IMPLEMENT `_assert_safe(query)` (THE FIREWALL)**: reject if the (lowercased, comment-stripped) query contains write clauses (`create|merge|set |delete|remove|drop|call db.create|call apoc.*` write procs) OR `sppaths` with a `maxlen` > 3 (parse the integer after `maxlen`; if absent on an SPpaths call, reject). Raise `ValueError("unsafe cypher: ...")`. This is the ONLY enforcement of the SPpaths wall now that candidates write raw Cypher.
- **IMPORTS**: `from starksearch.backends.falkordb import FalkorDBBackend`, `from starksearch.embedder import make_embedder`, `from graphretr_opt.config import load_config`, `from graphretr_opt.env.openai_client import OpenAIBudget`. (Harness is inside `starksearch/src` but imports the engine config/budget — both are on the worker PYTHONPATH: `opt_src` + `repo_root`, see `stark_subprocess_target.py:59-61`.)
- **GOTCHA**: the harness module is COPIED into the throwaway overlay (it lives under the base `src/`), so it must not import the editable service by absolute path that only exists post-materialize — import `StarkGraphSearchService` lazily inside `build_service` from the overlay package (`from stark_search.stark_graph_search_service import StarkGraphSearchService`). Keep `temperature`/model knobs sourced from `cfg`, never hardcoded.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -c "import sys; sys.path.insert(0,'../starksearch/src'); from stark_harness.qa_runner import CostSink, _assert_safe; _assert_safe('MATCH (n) RETURN n.id LIMIT 5'); print('ok')"`

### CREATE `starksearch/src/stark_harness/__init__.py` and `starksearch/src/stark_search/__init__.py`
- **IMPLEMENT**: empty package markers.
- **VALIDATE**: `test -f "starksearch/src/stark_harness/__init__.py" && test -f "starksearch/src/stark_search/__init__.py" && echo ok`

### CREATE `starksearch/src/stark_search/stark_graph_search_service.py`
- **IMPLEMENT**: `class StarkGraphSearchService` with `__init__(self, db, embedder, llm, **kwargs)` storing `self._db, self._emb, self._llm`. Port EVERY v7 helper from `starksearch/seeds/reasoning_first_v7.py` into methods: `_extract`, `_vector_search`, `_text_search`, `_llm_rerank`, `_reformulate`, `_graph_recall`, `_get_neighbors`, `_filter_nodes`, and the top-level `search(self, query) -> dict[int,float]` (the body of `reasoning_first_v7.py:260-356`, scoring levers + weights IDENTICAL). Replace `G.query(...)` → `self._db.cypher(...)`, `G.embed(...)` → `self._emb.encode(...)`, `G.llm(...)` → `self._llm(...)`, `G.get_text(...)` → a `_get_text` method built on `self._db.cypher`, `G.ntypes/labels/rel_types` → `self._db.ntypes/labels/rel_types`.
- **PATTERN**: `starksearch/seeds/reasoning_first_v7.py` (port 1:1). Class shape from `graphsearch/src/common/service/search/agentic_graph_traversal_search_service.py` (constructor-injected `graph`/`chat_model`).
- **IMPORTS**: pure Python + the injected clients only. **Real imports are now allowed** (no AST gate) — but the v7 logic needs none beyond builtins.
- **GOTCHA**: DELETE the phantom sandbox rules — you MAY now use normal `try/except ValueError`, real builtins, and module imports. Preserve the cosine distance→similarity conversion (`1.0 - score`, `reasoning_first_v7.py:116`) and the `LIMIT`/`maxLen<=3` discipline so the firewall never trips on the seed's own Cypher.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:../starksearch/src:src .venv/bin/python -c "from stark_search.stark_graph_search_service import StarkGraphSearchService as S; assert hasattr(S,'search'); print('ok')"`

### UPDATE `graphretr-demo/src/graphretr_opt/env/targets/_worker_stark.py`
- **IMPLEMENT**: Rewrite to mirror `_worker.py`. Read job `{overlay_dir, queries, falkor_cfg, cfg_overrides, opt_src, repo_root, query_concurrency}`. `sys.path.insert(0, overlay_dir)` FIRST, then `from stark_harness.qa_runner import CostSink, build_service, run_query`. Build `query_concurrency` isolated `(service, sink)` slots (reuse `_worker.py:58-90` sequential/parallel logic). Per query: `sink.reset()` → `pred = run_query(service, query)` → validate via the existing `_validate_pred` (keep `_worker_stark.py:34-48`) → `entry = {"pred": {str(k): v}, "cost": sink.snapshot(), "error": None}`. Print `{"results": {query: entry}}`.
- **PATTERN**: `_worker.py:32-90` (slots + `_one`); keep `_validate_pred` from current `_worker_stark.py:34-48`.
- **IMPORTS**: stdlib `json/sys/queue/traceback`, `concurrent.futures.ThreadPoolExecutor`.
- **GOTCHA**: drop `_build_graph` (no more in-worker `RetrievalGraph`) and the `_pin_query` hook — pinning was to stop `extract()` looping over node texts; with the editable service the prompt IS the candidate's, so it's obsolete. Per-query cost isolation now comes from per-slot `CostSink`, not getattr deltas.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_subprocess_target_smoke.py -q` (and the new fileset-seed smoke below).

### UPDATE `graphretr-demo/src/graphretr_opt/env/targets/stark_subprocess_target.py`
- **IMPLEMENT**: change `run(self, src, queries, timeout_s)` → `run(self, file_set, queries, timeout_s)`: `tmp = tempfile.mkdtemp(...)`, `overlay_dir = os.path.join(tmp, "src")`, `file_set.materialize(overlay_dir)`, put `overlay_dir` in the job, `try/finally: shutil.rmtree(tmp)`. Keep `_spawn`/`_kill`/`_parse`/`_all_error` (pred parsing at `:95-109`) unchanged. Add `query_concurrency` ctor arg passed into the job.
- **PATTERN**: `subprocess_target.py:58-75` (materialize + finally-cleanup). Keep STaRK's pred-shaped `_parse` (do NOT switch to `SearchResult`/`CostMeter`).
- **GOTCHA**: `file_set.materialize` requires `dest_dir` not pre-exist (`file_set.py:139`); materialize into `tmp/src`, not `tmp`.
- **VALIDATE**: covered by the fileset-seed smoke + parity test.

### UPDATE `graphretr-demo/src/graphretr_opt/reward/subprocess_reward.py`
- **IMPLEMENT**: confirm/adjust `StarkRewardAdapter.score(fn, idxs, src=..., ...)` forwards the FileSet (`src`) into `self._target.run(src, queries, batch_timeout)` (`:51`). Since `src` is now a FileSet, no signature change is needed — but verify `code_complexity(src)` (`:100`) handles a FileSet: pass `src.primary_file` source or `0.0` (the AST complexity cap is OFF for this target — see `_search_cfg`). Prefer `code_complexity(src.src_of(src.primary_file)) if hasattr(src,"overlay") else code_complexity(src)`.
- **PATTERN**: `subprocess_reward.py:35-109`; `_search_cfg` sets `gate_max_complexity=0.0` (`campaign.py:266`).
- **GOTCHA**: `code_complexity` (in `reward/objectives.py`) expects a string; a FileSet will break it. Guard as above.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests/ -q -k "stark or reward"`

### UPDATE `graphretr-demo/src/graphretr_opt/config.py`
- **IMPLEMENT**: add fields near `:103-112`: `stark_src: str = "starksearch/src"`, `stark_editable_files: tuple = ("stark_search/stark_graph_search_service.py",)`. Add property `stark_src_abs` mirroring `graphsearch_src_abs` (`:157-159`, resolves against `repo_root`). Handle `list→tuple` for `stark_editable_files` in `load_config` (mirror `:240-241`).
- **PATTERN**: `config.py:108-112, 157-163, 240-241`.
- **GOTCHA**: `repo_root` = parent of `graphretr-demo` (`:148-152`) = `Demo 1/`; `starksearch/src` resolves there. Keep `falkor_*`/`graph_name` as the STaRK FalkorDB cfg (already present `:22-24`).
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -c "from graphretr_opt.config import load_config; c=load_config(); print(c.stark_src_abs, c.stark_editable_files)"`

### REFACTOR `graphretr-demo/src/graphretr_opt/reward/evaluator.py` → move `QUALITY_KEYS`, then DELETE `RewardModel`
- **IMPLEMENT**: move `QUALITY_KEYS` (`evaluator.py:20`) into `starksearch/reward/__init__.py` (or `reward/objectives.py`) and update the import in `subprocess_reward.py:20`. Delete `class RewardModel` and the `from graphretr_opt.env.errors import SandboxError` it needs. Keep the file only if something else lives there; otherwise delete it.
- **PATTERN**: `subprocess_reward.py:19-20` (the only importer of `QUALITY_KEYS`).
- **GOTCHA**: `grep -rn "RewardModel\|from .evaluator\|reward.evaluator" graphretr-demo/src graphretr-demo/tests` first; update every importer. The parity test imports the adapter, not `RewardModel`.
- **VALIDATE**: `cd graphretr-demo && grep -rn "RewardModel" src tests | grep -v "StarkRewardAdapter" || echo "no RewardModel refs"`

### UPDATE `graphretr-demo/src/graphretr_opt/campaign.py::_boot_function` (and helpers)
- **IMPLEMENT**: rewrite `_boot_function` (`:61-112`) to mirror `boot_search`: (1) `sys.path` repo_root; (2) `from starksearch.qa import Substrate`; (3) build the STaRK `StarkSubprocessTarget` with `falkor_cfg`, `cfg_overrides={"root": cfg.root}`, `opt_src_dir=cfg.opt_src_abs`, `repo_root=cfg.repo_root`, `query_concurrency=cfg.query_concurrency`; (4) `self.reward = StarkRewardAdapter(self.substrate, target, cfg.crash_frac_limit, default_timeout_s=cfg.probe_timeout_s)`; (5) `self.seed = FileSet.from_base(cfg.stark_src_abs, cfg.stark_editable_files)`; (6) `self.sandbox = NullSandbox()`; (7) re-derive gate knobs for the larger service file (set `gate_max_complexity=0.0` — mirror `_search_cfg`'s complexity-off, keep STaRK's `gate_metric="recall@20"`, `gate_size`, meta-holdout). Remove the in-process `RetrievalGraph` construction (`:97-98`). Update `_make_mutator` (`:114-123`) to pass a static `STARK_DOMAIN_NOTE` (the primitive doc, derived from `graph.py:671-781` describe text) and `safe_builtins={}`. Update `_seed_program` (`:125-127`) to return the FileSet (or inline into boot). Make `optimize/stage0/final/ablate` pass the FileSet seed (they already call `self.reward.score(..., src=p.src)`; `FileSet.src` returns self — `file_set.py:67-75` — so call sites are unchanged).
- **PATTERN**: `boot_search` (`:161-233`), `_make_search_mutator` (`:316-325`), `_search_cfg` (`:235-278`).
- **GOTCHA**: STaRK keeps its own substrate/reward/metric (recall@20, NOT mcq_accuracy) — do NOT copy `_search_cfg`'s `gate_metric="mcq_accuracy"` blend. Only borrow the complexity-cap-off + (optionally) minibatch sizing. `final_test`/`ablate` (`campaign.py:159+` onward) save the best via `artifact.save(path)` — `FileSet.save` writes the primary file + `.files/` tree (`file_set.py:148-163`), so `best_search.py` still appears.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src GRAPHRETR_FAKE_TARGET=1 .venv/bin/python -c "from graphretr_opt.campaign import Campaign; from graphretr_opt.config import load_config; c=Campaign(load_config()).boot(); print(type(c.seed).__name__, c.seed.sha[:8])"` → expect `FileSet`.

### CREATE `graphretr-demo/tests/test_stark_firewall.py`
- **IMPLEMENT**: unit-test `ReadOnlyGraphClient._assert_safe` / `.cypher` with a fake backend (records the query, returns rows): assert `CREATE`/`MERGE`/`SET`/`DELETE`/`DROP` raise `ValueError`; `algo.SPpaths(... maxLen:4 ...)` and `maxLen:5` raise; `maxLen:3` and a plain `MATCH ... RETURN ... LIMIT 5` pass; rows truncated to the cap; `sink.db_queries` increments only on allowed queries.
- **PATTERN**: `tests/test_agentic_primitives.py` (cap-assertion style).
- **GOTCHA**: case-insensitive + comment-stripped matching; test lowercase, uppercase, and `// comment` variants.
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_stark_firewall.py -q`

### CREATE `graphretr-demo/tests/test_stark_fileset_seed.py`
- **IMPLEMENT**: `FileSet.from_base(cfg.stark_src_abs, cfg.stark_editable_files)` seeds without error; `materialize` to a tmp dir yields an importable tree exposing `stark_search.stark_graph_search_service.StarkGraphSearchService`; `build_service` constructs it with a fake backend + fake llm (monkeypatched, no infra) and `service.search("x")` returns a `dict[int,float]` (drive the fakes to return canned rows).
- **PATTERN**: `tests/test_file_set.py` + `tests/test_subprocess_target_smoke.py` (offline fakes).
- **VALIDATE**: `cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_stark_fileset_seed.py -q`

### REMOVE the `G` DSL and dead sandbox
- **IMPLEMENT**: delete `starksearch/graph.py` (operator layer `:213-781`) — if `describe()` text is reused as `STARK_DOMAIN_NOTE`, copy it into a constant first. Delete `starksearch/primitives.py` after moving `Allowlists` (to harness data) + input helpers (into the service). Delete the v7-and-siblings seeds under `starksearch/seeds/` ONLY after the parity test is green on the FileSet seed (keep `reasoning_first_v7.py` as the porting reference until then). Delete `RewardModel`/`reward/evaluator.py` (Task 8). Remove the phantom sandbox docstring rules wherever they survive.
- **PATTERN**: deletions justified by `grep -rn` showing no live importers.
- **GOTCHA**: `grep -rn "starksearch.graph\|RetrievalGraph\|starksearch.primitives\|import primitives" graphretr-demo starksearch` BEFORE deleting; the worker/campaign no longer import them after Tasks 4/9. The STaRK reward still imports `starksearch.qa` (Substrate) + `starksearch.reward` (Evaluator) — keep those.
- **VALIDATE**: `cd graphretr-demo && grep -rn "RetrievalGraph\|starksearch.primitives" src ../starksearch/src tests | grep -v "test_\|\.md:" || echo "clean"`

### UPDATE `graphretr-demo/tests/test_stark_parity.py`
- **IMPLEMENT**: point `SEED_PATH` (`:30`) at the new service file; the test must drive `StarkRewardAdapter → StarkSubprocessTarget → _worker_stark` with the FileSet seed and reproduce `golden/stark_parity_v7.json` ranking. If the ported service is byte-faithful the ranking is identical; only re-record the golden if you can show (diff of retrieved id order) the change is an intended numeric-noise difference, not a logic change.
- **PATTERN**: existing `test_stark_parity.py` (opt-in `STARK_PARITY=1`, live FalkorDB + funded key).
- **GOTCHA**: ranking equality is the invariant; raw scores drift ~1e-5 (live embeddings). Keep that tolerance.
- **VALIDATE**: `cd graphretr-demo && STARK_PARITY=1 PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_stark_parity.py -q` (needs infra; otherwise it skips and the offline suite must still pass).

---

## TESTING STRATEGY

Framework: `pytest` (`pyproject.toml:10-15`, `pythonpath=["src",".."]`). Live-infra tests are opt-in via env flags and skip cleanly offline (pattern: `test_stark_parity.py:21`).

### Unit Tests
- Firewall (`test_stark_firewall.py`): write/SPpaths rejection, row cap, read-only pass — no infra.
- FileSet seed (`test_stark_fileset_seed.py`): seed/materialize/import/construct with fakes — no infra.
- Harness clients: `CostSink` accounting, `LlmClient` cache hit returns without billing (monkeypatch budget).

### Integration Tests
- Worker smoke (extend `test_subprocess_target_smoke.py`): materialize the FileSet seed, run the STaRK worker on 1–2 queries against a fake backend/LLM, assert a `{pred,cost,error}` map back.
- Parity (`test_stark_parity.py`, `STARK_PARITY=1`): full real stack reproduces golden ranking for the v7-derived service.

### Edge Cases
- Candidate emits an unsafe query (`SPpaths maxLen:5`, a write) → `ValueError` in `db.cypher` → caught → scored as a per-query miss, optimizer survives.
- Candidate fails to import / wrong `search` signature → worker top-level `{"error",...}` → all queries miss (`subprocess_target.py:97-99`).
- Subprocess hangs → wall-clock kill of the process group (`stark_subprocess_target.py:84-93`).
- Empty/`None` pred, non-int keys → `_validate_pred` raises → per-query miss.
- `query_concurrency>1`: each slot has its own `CostSink`, metering never crosses queries.

---

## VALIDATION COMMANDS

Run from `graphretr-demo/`. Execute every command for zero regressions.

### Level 1: Syntax & Style
```
cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pyflakes src/graphretr_opt ../starksearch/src 2>/dev/null || PYTHONPATH=..:src .venv/bin/python -m py_compile src/graphretr_opt/env/targets/_worker_stark.py ../starksearch/src/stark_search/stark_graph_search_service.py ../starksearch/src/stark_harness/qa_runner.py
```

### Level 2: Unit Tests
```
cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_stark_firewall.py tests/test_stark_fileset_seed.py tests/test_file_set.py -q
```

### Level 3: Integration Tests
```
cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_subprocess_target_smoke.py -q
# full offline suite (live tests self-skip):
cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m pytest tests -q
```

### Level 4: Manual Validation
```
# boot the STaRK path offline and confirm the seed is a FileSet:
cd graphretr-demo && PYTHONPATH=..:src GRAPHRETR_FAKE_TARGET=1 .venv/bin/python -c "from graphretr_opt.campaign import Campaign; from graphretr_opt.config import load_config; c=Campaign(load_config()).boot(); print('seed:', type(c.seed).__name__, 'reward:', type(c.reward).__name__)"
# with live FalkorDB + key, a 1-step smoke:
cd graphretr-demo && PYTHONPATH=..:src .venv/bin/python -m graphretr_opt.cli optimize --steps 1 --campaign-name stark_fileset_smoke
```

### Level 5: Parity (live infra)
```
cd graphretr-demo && STARK_PARITY=1 PYTHONPATH=..:src .venv/bin/python -m pytest tests/test_stark_parity.py -q
```

---

## ACCEPTANCE CRITERIA

- [ ] The STaRK candidate is a `FileSet` over `starksearch/src`; `Campaign().boot()` (target=function) yields `type(seed).__name__ == "FileSet"`.
- [ ] `StarkGraphSearchService(db, embedder, llm).search(q)` returns `dict[int,float]`; built by `build_service` with injected, metered clients.
- [ ] The SPpaths/write firewall lives in the DB proxy; an unsafe candidate query raises and is scored as a miss (test proves writes + `maxLen>=4` rejected).
- [ ] `graph.py` operator layer, `primitives.py`, in-process `RewardModel`/AST sandbox path, and the phantom seed rules are deleted; `grep` shows no live importers.
- [ ] The parity test reproduces the golden ranking for the ported service (or golden re-recorded with a justified, numeric-noise-only diff).
- [ ] Offline `pytest tests -q` is green; no regressions in `graph_search` path (`test_file_set.py`, `test_subprocess_target_smoke.py` still pass).
- [ ] Mutator edits the STaRK service file with no mutator code change (artifact-agnostic path exercised).

## COMPLETION CHECKLIST

- [ ] All tasks completed in order, each validation passed immediately
- [ ] Full offline suite passes; live parity passes (or golden re-recorded with rationale)
- [ ] No `RetrievalGraph`/`primitives`/`RewardModel` references remain
- [ ] Manual boot shows FileSet seed + StarkRewardAdapter
- [ ] Firewall closes the SPpaths/write hole on the raw-Cypher path
- [ ] Acceptance criteria all met

---

## NOTES

**Why reuse `_boot_function`/target=function rather than a new `stark_search` target.** The STaRK path already wires `StarkRewardAdapter → StarkSubprocessTarget`; only the artifact (string→FileSet) and the worker (RetrievalGraph→harness) change. Reusing the path keeps `optimize/stage0/final/ablate` and the parity test intact and minimizes surface. (If a clean dual-name CLI is later wanted, add `optimize-stark-search` mirroring `optimize-search` — not required here.)

**Why the firewall must move into the proxy.** On v7 the candidate writes raw Cypher through `G.query`; the SPpaths cap was only in `RetrievalGraph.shortest_path`, which the candidate doesn't call — so the wall is currently bypassable and can peg the SHARED FalkorDB. The proxy is the one chokepoint every candidate query passes through. This is the single behavioral hardening in an otherwise behavior-preserving refactor.

**Metering parity.** The STaRK cost axes (`db_queries`, `llm_calls`, `rerank_items`, `latency_s`) previously came from getattr deltas on `RetrievalGraph` (`_worker_stark.py:93-116`). They now come from the per-slot `CostSink` written by the proxies — same axes, same MetricVector (`subprocess_reward.py:94-104`), so the gate/Pareto behavior is unchanged.

**Trade-off accepted.** The `LlmClient` reproduces `G.llm`'s disk cache + budget ceiling verbatim; the per-operator caches (extract/rerank/judge/...) in `graph.py` are NOT reproduced because the editable service owns those operators now — caching is by `(model,system,user)` at the one `llm()` chokepoint, which subsumes them.

**Confidence Score**: 8/10 for one-pass success. The engine `FileSet`/subprocess/mutator machinery already exists and is proven by `graph_search`; the main risks are (a) exactly preserving v7 ranking through the port (mitigated by the parity test) and (b) the firewall's Cypher parsing being neither too strict (tripping the seed's own `LIMIT`/`maxLen:3` queries) nor too loose — both have dedicated tests.
