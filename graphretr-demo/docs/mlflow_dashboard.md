# MLflow dashboard — the auditable views (Phase E)

No new infrastructure: this curates the views the self-hosted MLflow UI
(`http://127.0.0.1:5000`, experiment `graphretr-opt`) already renders, now that
the runs log the missing axes (Phases A–D). Everything below is reproducible by a
reviewer from the same store; nothing here is asserted only in prose.

## What every optimize run now carries

| field | where it shows | logged by |
|---|---|---|
| `config_hash` | param **and** run tag | `config.dump_resolved_config` (Phase A) |
| `git_sha` | param | `campaign._git_sha()` |
| `approach`, `strategy` | run tags | `campaign.optimize` `set_tags` (Phase D) |
| `best_quality_recall_at_20` / `_hit_at_1` / `_mrr` | per-step metric charts | `fast_loop` `log_vector("best_", …)` |
| `val_quality_*` | per-step (candidate) charts | `fast_loop` `log_vector("val_", …)` |
| `tokens_accepted` / `tokens_rejected` | per-step charts | `fast_loop` + `step_cost_delta` (Phase B) |
| `usd_step` / `usd_cumulative` / `usd_vs_ceiling` | per-step charts | Phase B |
| `calls_<model>_step` | per-step charts | Phase B |
| `accepted_total` / `steps_run` / `usd_total` | run summary metrics | Phase B (feed `$/accepted-edit`) |
| `resolved_config.yaml`, `lineage.jsonl`, `seed_vs_best.diff`, `reflection_*.md`, `best_search.py` | artifact viewer | `tracker.log_artifacts(run_dir)` |

## Curated views to build in the UI (one-time, saved per experiment)

1. **Table columns** — open the experiment table, *Columns* selector, pin in order:
   `Tags ▸ approach`, `Tags ▸ config_hash`, `Params ▸ git_sha`,
   `Metrics ▸ best_quality_recall_at_20`, `Metrics ▸ best_quality_hit_at_1`,
   `Metrics ▸ best_quality_mrr`, `Metrics ▸ usd_total`,
   `Metrics ▸ accepted_total`, `Metrics ▸ steps_run`.
   This is the "what changed × what it cost" row at a glance.

2. **Compare run3/run4/run5** — select the three rows → *Compare*. You get:
   - the **param diff** panel (highlights exactly which config fields differ —
     pair this with `config_hash` to confirm a real change vs. noise);
   - overlaid **metric history** charts. Add `best_quality_recall_at_20` and
     `best_quality_hit_at_1` (gate trajectory) and, on the same compare,
     `usd_cumulative` (spend trajectory) — gate-vs-cost on one screen.

3. **`$/accepted-edit`** — not a stored metric; read it as
   `usd_total ÷ accepted_total` from the table, or run
   `scripts/rollup.py` which computes the column directly (Markdown + CSV, and
   logs it back as the `rollup` run's artifacts).

4. **Per-step cost split** — open a single optimize run → *Metrics* →
   chart `tokens_accepted` vs `tokens_rejected` over `step`. The rejected area is
   spend that bought nothing; the accepted area is spend that moved the gate.

5. **Artifacts, inline** — open any run → *Artifacts*. Confirm these render in the
   browser without download: `resolved_config.yaml` (the exact effective config —
   recompute its hash to match the `config_hash` tag), `lineage.jsonl` (one row
   per step, parent→child sha chain, accepted *and* rejected), `seed_vs_best.diff`,
   `reflection_*.md`.

6. **Filter by approach / config_hash** — experiment table search box, e.g.
   `tags.approach = "run5-rerank"` or `tags.config_hash = "<hash>"` to isolate one
   line of work or one exact configuration across runs.

## Cross-run rollup

`PYTHONPATH=$PWD/src .venv/bin/python scripts/rollup.py` →
`runs/rollup/rollup.md` + `rollup.csv`, and a `rollup` MLflow run holding both as
artifacts. Columns: `run · approach · config_hash · git_sha ·
gate(recall@20/hit@1/mrr) · held-out(recall@20) · $/run · $/accepted-edit ·
steps · stopped_reason`. Held-out is joined from the matching
`final-test-report` run by the `campaign` param.

## Phase F — OpenAI tracing (optional, gated, OFF by default)

`mlflow.openai.autolog()` captures `G.extract` / `G.llm_rerank` calls as spans in
MLflow's trace UI. Enabled only when `MLFLOW_TRACE_OPENAI=1` and only on the
low-volume entrypoints (`stage0`, `final --test-n`) — **never** on `optimize`,
where span cardinality on a 33k-request campaign would balloon storage. See
`tracking/mlflow_tracker.maybe_enable_openai_tracing`.
