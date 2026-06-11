---
title: "Implementation plan — Stage-1 optimizer (verified against live env + sources)"
source: [Setupplan.md, handoffplan.md]
created: 2026-06-10
updated: 2026-06-10
tags: [plan, demo, optimizer, implementation, project]
connects: [Setupplan, handoffplan]
---

# Implementation plan — Stage-1 optimizer loop

Everything below is bound to **verified facts**: the live environment on this box, the installed `stark-qa` 1.1.0 / `falkordb` 1.6.1 sources, FalkorDB server source (`proc_vector_query.c`), and the actual `microsoft/SkillOpt` repo (`clip.py`, `trainer.py`, `reflect.py`, `default.yaml`). No guessed APIs.

## 0. Current environment state (audited 2026-06-10)

| component | state |
|---|---|
| Project root | `Demo 1/graphretr-demo/` — was moved from `/home/developer/graphretr-demo`; a **compat symlink** now exists at the old path (use it in scripts: no spaces, README stays valid) |
| FalkorDB | container `graphretr-falkordb` up, `127.0.0.1:6380`, graph `prime` |
| ETL | `etl_prime.py` **re-running now** (graph volume was empty); embeddings done, node load in progress → edges next. Idempotent; wait for completion + `smoke_test.py` PASS before building on it |
| MLflow | **fixed** — old server died pointing at the pre-move path; restarted via symlink path, `/health` → OK on `127.0.0.1:5000` |
| LLM access | **no `ANTHROPIC_API_KEY` in env**; `claude` CLI 2.1.170 is installed and authenticated → mutator backend options: (a) `claude -p` subprocess (zero config, default), (b) `anthropic` SDK if a key is provided |
| Python | `.venv` (3.11.15), all needed packages pinned in `requirements.txt` |

## 1. Design principles

- **One package, seven small modules, three entry scripts.** No framework, no plugin system, no abstract base classes. Each module has one responsibility and a one-line contract.
- **The mutable artifact is a file on disk** (`programs/`): the optimizer's whole product is readable as plain Python + a diff history. Everything else is frozen infrastructure.
- **Hard walls, not discipline:** the loop process physically cannot read the test split (separate entry script), candidate code physically cannot write to the DB (`ro_query` only) or import anything (AST gate + empty namespace).
- **Dict-shaped scores everywhere** (`{'recall@20': .., 'hit@1': .., 'mrr': ..}`) — the GEPA Stage-2 seam.

## 2. Module layout

```
graphretr-demo/
├── etl_prime.py, smoke_test.py, load_check.py   # existing Stage-1a (untouched)
├── optimizer/
│   ├── config.py        # ~30 loc  all knobs in one dataclass; env overrides
│   ├── primitives.py    # ~180 loc RetrievalGraph: 7 read-only capped primitives + memo cache
│   ├── data.py          # ~60 loc  QA load, splits→python ints, frozen val-gate subsample
│   ├── scorer.py        # ~60 loc  score(fn, idxs) -> metric dict (STaRK Evaluator)
│   ├── sandbox.py       # ~70 loc  AST safety gate + exec-compile of candidate source + probe cascade
│   ├── mutate.py        # ~120 loc reflection prompt, LLM call (claude-CLI or SDK), edit-budget clip
│   └── loop.py          # ~150 loc the campaign: rollout→reflect→gate→accept/reject + MLflow
├── programs/
│   ├── seed.py          # the naive vector-only search (frozen copy)
│   └── runs/<campaign>/ # step_NNN.py candidates + accepted/ + rejected buffer.json
└── scripts/
    ├── run_stage0.py    # one-shot frontier rewrite vs seed (headroom check)
    ├── run_campaign.py  # the loop (NEVER touches test)
    └── run_final_test.py# locked test split, scored once, separate MLflow run
```

~670 lines of new code total. No other files.

## 3. Verified contracts the code must honor

**STaRK (`stark_qa` 1.1.0, from source):**
- `qa[i]` → `(query: str, q_id: int, answer_ids: list[int], None)`. Positional `i` ≠ `q_id`.
- `qa.get_idx_split()` → `dict[str, torch.LongTensor]` of **positional** indices. Prime: train 6,161 / val 2,240 / test 2,800.
- `Evaluator(skb.candidate_ids)` — plain `list[int]`, all 129,375 ids.
- `ev.evaluate(pred_dict, answer_ids, metrics)` — `pred_dict: Dict[int, float]` (partial ranking fine; unscored candidates ranked last), `answer_ids` **must be `torch.LongTensor`** (wrap the list!), returns `Dict[str, float]`. `'hit@1'/'recall@20'/'mrr'` are valid names.
- Gotchas: empty `pred_dict` crashes (`min()` of empty) → sandbox cheap-reject guards; predicted id > max(candidate_ids) → IndexError → primitives only ever return real node ids, so impossible by construction.

**FalkorDB (client 1.6.1 + server source):**
- All primitive queries via `g.ro_query(q, params=dict, timeout=ms)` → `GRAPH.RO_QUERY` (server rejects writes — the read-only wall is free).
- Vector: `CALL db.idx.vector.queryNodes($label,'embedding',k, vecf32($vec)) YIELD node, score` — **score is cosine DISTANCE (0..2, lower=closer), pre-sorted ascending**. The docs' `ORDER BY score DESC` example is wrong. Primitives return `similarity = 1 - score` so programs always rank descending.
- Vector/fulltext indexes are **per-label** (10 of them) — `vector_search` loops labels (or takes a `label=` arg) and merges top-k client-side. Label is a proc string arg → parameterizable.
- Relationship type in `-[:T]->` patterns is **not parameterizable** → interpolate from the 18-type allowlist (validated), everything else as `$params`.
- `k_hop_expand` uses `algo.bfs(start, $depth, $relType-or-null)` (dedups during traversal — var-length `[*1..k]` path-enumerates and explodes on biomedical hubs). `shortest_path` uses `algo.SPpaths` (rel-type + direction + maxLen filters).
- Set `TIMEOUT_MAX` + `QUERY_MEM_CAPACITY` once via `GRAPH.CONFIG SET` at campaign start; every `ro_query` carries `timeout=2000`.

**SkillOpt (actual repo defaults → our port):**
| SkillOpt | ours |
|---|---|
| edit budget `L` 4→2 cosine, clip never rejects | constant **L=4**; count diff hunks (difflib opcodes) of candidate vs incumbent; over budget → 1 retry with feedback → else skip step |
| rollout batch 40, reflect minibatch 8 | rollout batch **24** train queries, reflection sees the ≤8 worst misses |
| strict `>` gate, ties rejected, hash-cached scores | same: gate on `recall@20` over the frozen val subsample; `sha256(source)` → score cache |
| reject buffer `{score_before, score_after, rejected_edits}`, all fed back, epoch-evicted | same fields + the unified diff; **last-8** fed into the prompt (flat loop, no epochs); full buffer persisted as artifact |
| one combined diagnose+edit call, reasoning model | same: single call to `claude` (Opus-class), no temperature knob needed |

## 4. Component specs

### config.py
One frozen dataclass: falkor host/port/graph, mlflow uri, `GATE_SIZE=200`, `GATE_SEED=42`, `ROLLOUT_BATCH=24`, `MAX_EDITS=4`, `STEPS=30`, `QUERY_TIMEOUT_MS=2000`, caps (`MAX_FANOUT=200`, `MAX_K=3`), mutator backend (`cli`|`sdk`) + model name. Env-var overrides (`FALKOR_PORT` etc., matching the ETL's existing convention).

### primitives.py — `RetrievalGraph` (the frozen instruction set)
Seven methods, each = one parameterized read-only Cypher + explicit cap; **every method returns node-id-keyed data, never raw graph objects**:
```python
vector_search(text, k=20, label=None)        -> list[(node_id, similarity)]   # 10-label fan-out, merged
get_neighbors(ids, rel_type=None, direction='out', limit=50) -> list[(src, rel, dst)]
k_hop_expand(ids, k=2, rel_type=None, max_nodes=200) -> list[node_id]          # algo.bfs
filter_nodes(ids, ntype=None, text_contains=None, limit=200) -> list[node_id]
shortest_path(a, b, max_len=4)               -> list[node_id]                  # algo.SPpaths
rank_by_text(ids, query_text, top=20)        -> list[(node_id, similarity)]    # embed query once, cosine vs node embeddings fetched by id
get_text(ids, limit=50)                      -> dict[node_id, str]             # for text filtering
```
- Rel-type/label args validated against the allowlists read from the graph at startup (`db.labels()`, `db.relationshipTypes()`).
- **Memo cache**: `dict[(method, frozen_args)] -> result` + a query-embedding cache (`text -> vec`). The gate re-scores the same 200 queries every step — this keeps the loop LLM-bound, not DB-bound.
- The embedder (`all-MiniLM-L6-v2`, the same model the ETL used) loads once here.

### data.py
Loads qa + skb once; converts split tensors to `list[int]`; builds the **frozen gate subsample** — 200 val indices sampled with `GATE_SEED`, written to `programs/runs/<campaign>/gate_idxs.json` on first run, **read back (never resampled) thereafter**. Exposes `train_idxs`, `gate_idxs`. `test_idxs` lives behind `get_test_idxs_I_KNOW_THIS_IS_FINAL()` — called only by `run_final_test.py`.

### scorer.py
```python
def score(search_fn, idxs) -> dict:   # {'recall@20':…,'hit@1':…,'mrr':…}
```
Wraps each call: `pred = search_fn(query, G)`; guards (non-empty dict, int keys, float values) → per-query `ev.evaluate(pred, torch.LongTensor(answer_ids), ['hit@1','recall@20','mrr'])`; mean per metric. Also returns per-query rows on request (`return_rows=True`) — that's what rollout uses to find the worst misses. A candidate raising on >10% of queries scores 0 (logged as `crashed`).

### sandbox.py
- **AST gate** (allowlist, ~25 lines): module must define exactly `def search(q, G)`; forbidden anywhere: `import`, `exec/eval/compile/open/__*__` attribute access, names outside `{q, G, local vars}` + safe builtins (`len,sorted,set,dict,list,min,max,sum,enumerate,zip,range,abs,float,int,str`).
- **Compile**: `exec` in a namespace containing only those builtins; returns the `search` callable.
- **Probe cascade** (OpenEvolve pattern): before any 200-query gate, run the candidate on 3 fixed train queries — must return a non-empty `dict[int,float]` in <10s each. Fail → candidate rejected at zero gate cost, error text goes into the reflection buffer.

### mutate.py
- Builds the reflection prompt exactly per Setupplan §"reflection prompt": current source, primitive signatures+semantics, the ≤8 worst rollout failures (query text, gold ids **with their node texts**, retrieved ids, missed gold ids), last-8 rejected-edit buffer entries.
- One LLM call → expects fenced full function body + per-edit one-liners. Backend `cli`: `claude -p --model opus` subprocess (works today, zero config); backend `sdk`: `anthropic` client (when a key lands in `.env`).
- **Edit-budget clip**: `difflib.SequenceMatcher.get_opcodes()` between incumbent and candidate source; count non-`equal` hunks; `>MAX_EDITS` → one retry telling it which budget it blew → still over → step skipped, logged.

### loop.py (`run_campaign.py` drives it)
```python
G, qa = boot()                                  # primitives + data + mlflow run
prog_src = read(programs/seed.py); best = score(compile(prog_src), gate_idxs)
buffer = []                                     # rejected edits
for step in range(STEPS):
    rows  = score(prog, sample(train_idxs, 24), return_rows=True)      # rollout
    fails = worst_by_recall(rows)[:8]
    cand_src = llm_edit(prog_src, fails, buffer[-8:])                  # mutate + clip
    fn = sandbox.compile_and_probe(cand_src)                           # cascade
    s  = cached_score(fn, gate_idxs)                                   # sha256 cache
    if s and s['recall@20'] > best['recall@20']:
        prog_src, best = cand_src, s; save accepted/step_NNN.py        # ACCEPT
    else:
        buffer.append({'diff': …, 'score_before': best, 'score_after': s})  # REJECT
    mlflow: log s + best + accept flag, artifacts (cand_src, transcript, buffer.json)
write programs/runs/<campaign>/best_search.py
```
Deterministic seeds (`random.Random(GATE_SEED + step)` for rollout sampling) → reruns comparable.

### Entry scripts
- **run_stage0.py** — gate check before investing: score seed on gate set; one frontier one-shot rewrite (same prompt, no loop); score it. If the one-shot already closes most headroom, stop and re-scope (per Setupplan pitfall #1). Logged to MLflow as `stage0`.
- **run_campaign.py** — args: `--steps --campaign-name`; everything else from config.
- **run_final_test.py** — loads `best_search.py` + seed, scores both on the full test split **once**, separate MLflow run `final-test-report`. The only file that touches test.

## 5. Observability (MLflow, server already running)

- Experiment `graphretr-opt`; one run per campaign, `stage0` and `final-test-report` as separate runs.
- Params: all of config + mutator model + seed/campaign name + git sha.
- Per step: `val_recall@20`, `val_hit@1`, `val_mrr`, `best_recall@20`, `accepted` (0/1), `gate_cache_hit`, `probe_failed`, `llm_seconds`, `score_seconds`.
- Artifacts per step: `step_NNN.py`, `reflection_NNN.md` (prompt+response transcript), rolling `buffer.json`. End: `best_search.py` + `seed_vs_best.diff`.
- This is exactly demo exhibit #3: the curve, the gate accept/rejects, the buffer suppressing repeats.

## 6. Build order (each step has a hard verification gate)

1. **Wait for ETL** → `smoke_test.py` PASS (counts 129,375 / 8,100,498; ANN + 1-hop return ids; MLflow healthy). *Blocked until the running ETL finishes.*
2. **primitives.py** + 10-line pytest-style check: each primitive returns plausible ids on 2 known queries; `vector_search` similarity ∈ [0,1] descending; rel-type allowlist rejects junk.
3. **data.py + scorer.py** → score the seed on the gate set. Expect recall@20 roughly in the 0.1–0.2 band (well under the 0.381 reported ceiling — that's the headroom). Record the number.
4. **sandbox.py** → seed compiles through it; a malicious probe (`import os`) is rejected.
5. **run_stage0.py** → decision point: loop worth building? (expected: yes — one shot can't tune caps/thresholds against feedback).
6. **mutate.py + loop.py** → 3-step dry-run campaign (`STEPS=3`), verify MLflow curve + artifacts + one forced rejection lands in the buffer.
7. **Full campaign** (`STEPS=30`, ~1-2h wall-clock, LLM-bound) → `run_final_test.py` → the before/after numbers + diff = demo exhibits #1 and #2.

## 7. Risks / open items

- **LLM auth**: no API key on the box. Default backend = `claude -p` subprocess (CLI 2.1.170 authenticated). Drop an `ANTHROPIC_API_KEY` into `graphretr-demo/.env` to switch to the SDK backend (cleaner transcripts, token counts in MLflow).
- **Gate noise**: 200 queries → ±~0.02 std on recall@20; strict-`>` gate plus fixed subsample handles it, but expect some lucky accepts; the final test split is the honest arbiter.
- **8 GB RAM**: skb + qa + embedder + torch ≈ 2–3 GB; fine, but don't run the campaign while the ETL is still loading edges.
- **Hub-node blowups**: contained by `algo.bfs` + per-query `timeout=2000` + `QUERY_MEM_CAPACITY`; a timing-out candidate just scores poorly (and the error text feeds reflection) — it cannot take down the loop.
