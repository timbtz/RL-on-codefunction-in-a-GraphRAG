# Plan: Bring up real infra and run the optimizer against the live Neo4j KG

Goal: take the proven offline loop to a **real first run** — search service hits a
restored Neo4j KG, in-search + answerer LLM calls go through **OpenRouter**, the
mutator stays on the **Claude CLI** (subscription), and everything logs to a real
**MLflow** backed by SQLite with a browsable UI.

Status going in (verified 2026-06-24):
- `.venv` = Python **3.14.4**. Full `mlflow 3.14.0` (+ SQLAlchemy/Alembic), `pandas 2.3.3`, `numpy 2.5.0`, `pyarrow 24`, `neo4j` installed.
- **Missing for the real path:** `langchain`, `langchain-core`, `langchain-neo4j`, `langchain-openai`, `langchain-anthropic`.
- Offline `--fake-target` step already runs end-to-end and logs to MLflow.
- Mutator uses `mutator_backend="cli"` (Claude CLI). It only flips to `sdk` if `ANTHROPIC_API_KEY` is set → **keep `ANTHROPIC_API_KEY` unset**.

---

## Workstream 1 — Python versioning

Decision: **stay on the existing 3.14 venv** (everything installed so far works;
`requires-python>=3.11` is satisfied) and install the real-path deps there. Fall
back to a dedicated 3.11/3.12 venv **only if** a langchain/pydantic-core wheel is
missing for 3.14.

Steps:
1. Install the real-path deps into `.venv` (use looser, source-of-truth pins from
   `graphsearch/requirements.txt`, not the heavy STaRK lock):
   ```
   .venv/bin/python -m pip install \
     "langchain-neo4j>=0.8.0" "langchain-openai>=1.1.9" \
     "langchain-anthropic>=1.3.4" "langchain-core>=1.2.12" "langchain>=1.2.10"
   ```
2. Verify imports on 3.14:
   ```
   .venv/bin/python -c "import langchain_neo4j, langchain_openai, langchain_core, neo4j; print('real-path imports OK')"
   .venv/bin/python -c "import sys; sys.path.insert(0,'graphsearch/src'); \
     from common.service.search.agentic_graph_traversal_search_service import AgenticGraphTraversalSearchService; print('service imports OK')"
   ```
3. **Fallback (only if a wheel fails):** `brew install python@3.11`, recreate the
   venv with `python3.11 -m venv .venv`, reinstall mlflow(full) + minimal subset +
   real-path deps. Document whichever interpreter we standardize on.
4. Note the side effects already in play (from the mlflow full install): `pandas`
   pinned to `2.3.3` (mlflow 3.14 doesn't support pandas 3.x yet), `cryptography`
   `48.0.1`. Leave as-is unless something breaks.

Acceptance: both verify commands above print OK on the chosen interpreter.

---

## Workstream 2 — MLflow real backend (drop the file-store workaround)

We now have full mlflow, so we no longer need `MLFLOW_ALLOW_FILE_STORE=true` or the
deprecated file store. Use **SQLite backend + artifact dir**, with an optional UI.

The tracker just calls `mlflow.set_tracking_uri(cfg.mlflow_url)`; `cfg.mlflow_url`
defaults to `http://127.0.0.1:5000`. Two supported modes:

- **Serverless (simplest):** point runs at the SQLite DB directly.
  ```
  export MLFLOW_URL="sqlite:///$PWD/graphretr-demo/mlflow.db"
  ```
- **With UI (recommended):** run a server (default tracking URL already matches it).
  ```
  cd graphretr-demo
  ../.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 &
  # then runs use the default cfg.mlflow_url=http://127.0.0.1:5000 (no env needed)
  ```

Steps:
1. Pick UI mode; start the server (background) before runs.
2. Stop passing `MLFLOW_ALLOW_FILE_STORE`. Optionally clean the dry-run's
   `graphretr-demo/mlruns/` file-store experiment metadata (artifacts can stay).
3. Re-run the offline `--fake-target --steps 1` once and confirm the run + metrics
   (`best_quality_mcq_accuracy`, `retrieval_hit`, cost axes) + artifacts (recap,
   lineage, diff, candidate code, reflection) appear **in the MLflow UI**.

Acceptance: the fake run shows up at `http://127.0.0.1:5000` with metrics, params,
and the recap/lineage/code artifacts attached.

Optional cleanup (separate, not blocking): the shared `FastLoop` still logs
STaRK-vocabulary metrics (`recall@20`, `recall@100`, `hit@1/5`, `mrr`,
`rerank_items`) on the MCQ path — all 0 and noisy. Consider gating these behind the
target so the MCQ UI shows only `mcq_accuracy`/`retrieval_hit` + cost.

---

## Workstream 3 — Neo4j restore + bring-up

The backup `neo4j/multi.video-2026-04-21T06-09-06.backup` is an **Enterprise**
`.backup` (BZV2 + zstd). DB name encoded = **`multi.video`**. Restore needs
Enterprise `neo4j-admin`; use the official Enterprise Docker image (eval license).

Steps (Docker daemon must be up):
1. **Offline restore** into a named volume:
   ```
   docker run --rm \
     -v "$PWD/neo4j:/backups" -v neo4j_multi_data:/data \
     -e NEO4J_ACCEPT_LICENSE_AGREEMENT=eval \
     neo4j:enterprise \
     neo4j-admin database restore --from-path=/backups --overwrite-destination=true 'multi.video'
   ```
2. **Start** the server on that volume (choose a password):
   ```
   docker run -d --name neo4j-multi -p 7474:7474 -p 7687:7687 \
     -v neo4j_multi_data:/data \
     -e NEO4J_ACCEPT_LICENSE_AGREEMENT=eval -e NEO4J_AUTH=neo4j/<password> \
     neo4j:enterprise
   ```
3. **Register** the restored store in the system catalog:
   ```
   docker exec -it neo4j-multi cypher-shell -u neo4j -p <password> -d system \
     "CREATE DATABASE \`multi.video\` IF NOT EXISTS;"
   ```
4. **Verify** node counts, labels, and the fulltext index the service depends on:
   ```
   docker exec -it neo4j-multi cypher-shell -u neo4j -p <password> -d 'multi.video' \
     "MATCH (n) RETURN labels(n) AS l, count(*) ORDER BY count(*) DESC;
      SHOW INDEXES YIELD name,type WHERE type='FULLTEXT' RETURN name;"
   ```
   - The service hard-needs a fulltext index named **`ft_Entities`** over `:Entity`
     and `:Document` nodes with `id`+`text`. If missing, create it (props TBD from
     the verify output).

Runtime-verify gotchas (can't pre-check; adapt live): exact `neo4j-admin restore`
flag spelling for the image's version; the post-restore `CREATE DATABASE` vs
`START DATABASE` behavior; whether `multi.video` (dotted name) needs backticks
everywhere (it does in Cypher).

Acceptance: `multi.video` is online, node counts look like the video/transcript KG,
and `ft_Entities` exists (or is created).

---

## Workstream 4 — OpenRouter wiring (in-search + answerer; mutator stays Claude CLI)

OpenRouter is OpenAI-API-compatible: set `base_url=https://openrouter.ai/api/v1`,
`api_key=<OPENROUTER_KEY>`, and an OpenRouter model id. Both LLM construction sites
use bare `ChatOpenAI(model=..., temperature=0)` with no base_url, so they need a
small change to read base_url + key from env. Keep `provider="openai"` (OpenRouter
speaks the OpenAI dialect).

Two sites:
- `graphsearch/src/common/service/qa_eval/qa_runner.py:102` — the **search service**
  chat model (runs in the subprocess).
- `graphretr-demo/src/graphretr_opt/campaign.py:225` — the **MCQ answerer** (runs in
  the main process).

Steps:
1. **Env contract** — create `graphsearch/.env` (gitignored) and load it:
   ```
   # OpenRouter (in-search + answerer)
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   GRAPHSEARCH_SEARCH_MODEL=deepseek/deepseek-chat        # or google/gemini-2.5-flash-lite
   GRAPHSEARCH_ANSWERER_MODEL=deepseek/deepseek-chat
   # Neo4j
   GRAPHSEARCH_NEO4J_URL=bolt://localhost:7687
   GRAPHSEARCH_NEO4J_USER=neo4j
   GRAPHSEARCH_NEO4J_PASSWORD=<password>
   GRAPHSEARCH_NEO4J_DATABASE=multi.video
   ```
   Add `.env` loading at the optimizer entrypoint (cli/config) via `python-dotenv`
   (already a dep), or `set -a; source graphsearch/.env` before running.
   **Confirm `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are NOT set** (so the mutator
   stays on the Claude CLI and ChatOpenAI uses the OpenRouter key/base_url).

2. **Code change — make the OpenAI chat builder OpenRouter-aware.** In both sites,
   when `OPENROUTER_API_KEY`/base_url (or generic `*_BASE_URL`/`*_API_KEY`) are set,
   pass `base_url=` and `api_key=` to `ChatOpenAI`. Prefer a single shared helper
   (e.g. in `qa_runner`) so the answerer and the service share one construction
   path. `temperature=0` stays (reward stability).
   - The subprocess inherits `os.environ`, so the worker's `build_service` sees the
     same vars — just read them in `_build_chat_model`.
   - Pass the model ids through config (`search_model`, `answerer_model`) from the
     new env vars (`config.py` already maps `GRAPHSEARCH_*`/`ANSWERER_MODEL`; add
     `SEARCH_MODEL` + base_url/key fields as needed).

3. **Confirm model ids** on OpenRouter before the run (exact slugs): DeepSeek V3
   (`deepseek/deepseek-chat`) and Gemini 2.5 Flash Lite
   (`google/gemini-2.5-flash-lite`).

4. **Smoke test** one query through the real path (no optimizer loop) to prove the
   service talks to OpenRouter + Neo4j and returns `[doc:ID]` context:
   ```
   set -a; source graphsearch/.env; set +a
   .venv/bin/python -c "import sys; sys.path.insert(0,'graphsearch/src'); \
     from common.service.qa_eval.qa_runner import build_service, run_query; \
     svc=build_service({'url':'bolt://localhost:7687','username':'neo4j','password':'<pw>','database':'multi.video'}, \
       {'provider':'openai','model':'deepseek/deepseek-chat'}); \
     print(run_query(svc, 'Welche 2 AddOns laufen im IOT Devicegateway?')[:500])"
   ```

Acceptance: the smoke query returns real graph context (not an error/empty), routed
through OpenRouter.

---

## Workstream 5 — First real run + verify

1. With Neo4j up, `.env` sourced, MLflow server running, run a short real campaign:
   ```
   cd graphretr-demo
   PYTHONPATH=src ../.venv/bin/python -m graphretr_opt.cli optimize-search --steps 3 --campaign-name search-real-001
   ```
   (Start with `--steps 1`–`3` to validate cost/latency before a longer run.)
2. Confirm in MLflow: non-zero `retrieval_hit`/`mcq_accuracy`, per-question recap
   shows real retrieved context, cost axes populated from the subprocess meter.
3. Confirm the mutator ran via Claude CLI (`LLM calls by model: {claude-*}`), and
   in-search calls were billed to OpenRouter (not OpenAI).
4. Inspect `runs/search-real-001/seed_vs_best.diff` and `best_recap.json`.

Acceptance: a real first run produces a non-trivial baseline + at least one scored
candidate, fully logged, with retrieval actually exercising the KG.

---

## Open decisions / risks
- **Interpreter:** 3.14 (current) vs a 3.11 fallback — decided by whether langchain
  wheels resolve on 3.14 (Workstream 1 gate).
- **Search vs answerer model:** DeepSeek V3 and/or Gemini 2.5 Flash Lite — pick per
  cost/quality; they can differ (`search_model` vs `answerer_model`).
- **`multi.video` dotted DB name:** verify Neo4j accepts it as-is on restore +
  CREATE; backtick-quote everywhere.
- **`ft_Entities` index:** may or may not be inside the backup — create if absent.
- **STaRK metric noise** in MLflow on the MCQ path — optional cleanup, non-blocking.
