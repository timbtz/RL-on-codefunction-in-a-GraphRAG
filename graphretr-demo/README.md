# graphretr-demo — Stage 1a environment (STaRK-prime → FalkorDB + MLflow)

Standing environment for the graph-retrieval optimizer demo: STaRK **prime**
loaded into **FalkorDB** (graph + vector index), with an **MLflow** tracking
server running locally. This stage is **environment + data only** — no
optimizer, no retrieval-primitive library, no MLflow logging integration yet.

Project root: `/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo`
(quote it in shell commands — the path contains spaces).
All services bind to **127.0.0.1 only** (no public exposure).

---

## What's installed

- **OS / host:** Ubuntu (Hetzner VPS, 4 vCPU, 8 GB RAM, no GPU), Docker 27.0.3.
- **Python:** 3.11.15 venv at `.venv/` (created with `uv`; STaRK ecosystem is
  happiest on Python <3.12). Installed-package metadata says stark-qa only
  requires `>=3.8`, but 3.11 is used to stay clear of the numpy/PyG/rdkit
  edges — see deviations.
- **Key packages** (full pin list in `requirements.txt`):

  | package | version |
  |---|---|
  | stark-qa | 1.1.0 |
  | falkordb (client) | 1.6.1 |
  | sentence-transformers | 5.5.1 |
  | mlflow | 3.13.0 |
  | anthropic | 0.109.1 (installed for later, unused) |
  | openai | 2.41.0 (installed for later, unused) |
  | torch | 2.12.0+cpu (CPU-only build) |
  | numpy | 2.4.6 |

- **FalkorDB:** Docker image `falkordb/falkordb:latest` (Redis 8.2.3 engine),
  container `graphretr-falkordb`, data on named volume `graphretr-falkordb-data`.

---

## Services

| service | bind | container / process | notes |
|---|---|---|---|
| FalkorDB (Redis protocol) | `127.0.0.1:6380` | container `graphretr-falkordb` | `--restart unless-stopped` |
| FalkorDB browser UI | `127.0.0.1:3001` | same container | |
| MLflow tracking server | `127.0.0.1:5000` | venv `mlflow server` via `nohup` | sqlite backend |

### Start / stop

```bash
# --- FalkorDB ---
docker start graphretr-falkordb         # start
docker stop  graphretr-falkordb         # stop
docker exec  graphretr-falkordb redis-cli -p 6379 PING   # -> PONG (inside container)
redis-cli -h 127.0.0.1 -p 6380 PING                      # -> PONG (from host, if redis-cli present)

# --- MLflow ---  (started with nohup; pid in mlflow.pid)
cd "/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo"
# NOTE: must go through `python -m` — the venv's console-script shebangs
# point at the pre-move path, and Linux shebangs cannot contain spaces.
nohup .venv/bin/python -m mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri "sqlite:////home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo/mlflow.db" \
  --default-artifact-root "/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo/mlartifacts" \
  > mlflow.log 2>&1 &  echo $! > mlflow.pid
kill "$(cat mlflow.pid)"                 # stop
curl -s http://127.0.0.1:5000/health     # -> OK
```

> The first FalkorDB container in the briefing maps `6379`, but this host
> already runs another FalkorDB on `127.0.0.1:6379` (the agent harness) plus a
> Coolify PaaS stack — so this demo uses **6380 / 3001** to stay isolated.
> Override in any script via env: `FALKOR_HOST`, `FALKOR_PORT`, `GRAPH_NAME`.

---

## The FalkorDB graph

- **Graph name:** `prime`
- **Nodes:** 129,375 across 10 STaRK node-types. Every node carries:
  - a shared label `:Entity` (with a **range index on `id`** — makes edge
    `MATCH` by id fast), **plus**
  - a per-type label = the STaRK node-type, sanitized to a valid Cypher label
    (`gene/protein → gene_protein`, `molecular_function`, `effect/phenotype →
    effect_phenotype`, `biological_process`, `cellular_component`, `disease`,
    `drug`, `pathway`, `anatomy`, `exposure`).
  - properties: `{ id (int STaRK index), ntype (exact STaRK type string),
    text (skb.get_doc_info), embedding (384-d vecf32) }`.
- **Edges:** 8,100,498, directed `src(edge_index[0]) → dst(edge_index[1])`,
  typed by sanitized STaRK relation name (18 types, e.g. `ppi`, `target`,
  `indication`, `off_label_use`, `parent_child`, `associated_with`, ...).
- **Vector index:** cosine, dimension 384, on `embedding` — **one per per-type
  label**. Query with:
  ```cypher
  CALL db.idx.vector.queryNodes('gene_protein','embedding',20, vecf32($qvec))
  YIELD node, score RETURN node.id, score
  ```
- **Embeddings:** `all-MiniLM-L6-v2` (384-d, CPU, normalized), `max_seq_length`
  capped at 128 tokens for speed. Cached to `emb_cache.npy` (+ `.meta.json`)
  so the ETL never re-embeds.

---

## Stage-1a scripts (environment + data)

| script | purpose |
|---|---|
| `load_check.py` | Load STaRK prime in Python; print counts, node types, schema triples, a sample node doc, a sample query + gold answer_ids, split sizes. |
| `etl_prime.py`  | One-time, **idempotent** ETL of prime into FalkorDB (embed → nodes → indexes → edges). Skips if graph already full; rebuilds if partial; never re-embeds (cache). |
| `smoke_test.py` | PASS/FAIL: prime loads · FalkorDB counts match · ad-hoc ANN vector search returns ids · ad-hoc 1-hop neighbors return ids · MLflow `/health`. |

```bash
cd "/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo"
.venv/bin/python load_check.py
.venv/bin/python etl_prime.py     # one-time; safe to re-run (idempotent)
.venv/bin/python smoke_test.py
```

## Stage-1 optimizer (`src/graphretr_opt/`)

An optimizer service that **rewrites a `search(q, G)` Python program** over the
frozen graph, with the LLM frozen. Two halves with a hard wall:

```
src/graphretr_opt/
├─ env/            IMMUTABLE: backends/{base,falkordb,neo4j} · retrieval_graph
│                  (the 7-primitive DSL) · primitives (validation/allowlists)
│                  · cache · embedder · sandbox (AST gate + run→RunStats)
├─ data/           substrate (qa/skb, splits, frozen gate, Evaluator) · loader_etl
├─ artifact/       program (SearchProgram: mutable layer-2 src+diff) · seeds/
├─ reward/         objectives (MetricVector — never scalarized) · evaluator
│                  (RewardModel) · pareto (dominance + MAP-Elites)
├─ optimizer/      mutator · edit_budget · rejected_buffer · gate · fast_loop
│                  · momentum/slow_loop/scheduler  (Stage-2 seams)
├─ agents/         single (SingleCoder) · team  (Stage-2 seam)
├─ tracking/       mlflow_tracker
├─ gepa_adapter.py  Stage-2 seam (the one new file Stage 2 needs)
├─ campaign.py      orchestrator — wires the halves
└─ cli.py           stage0 | optimize | final
configs/           campaign.yaml + strategies/*.yaml
tests/             test_smoke.py — primitives + sandbox + unit seams (20 checks)
runs/<campaign>/   per-campaign artifacts (frozen gate_idxs.json, steps, buffer)
```

Run it (config from `configs/campaign.yaml` + env vars; `src/` is on the path
via `pyproject.toml`/`PYTHONPATH`):

```bash
cd "/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo"
export PYTHONPATH="$PWD/src:$PYTHONPATH"           # or: source .env.sh
.venv/bin/python -m tests.test_smoke               # 20/20 structural checks
.venv/bin/python -m graphretr_opt.cli stage0       # seed vs one-shot headroom
.venv/bin/python -m graphretr_opt.cli optimize --steps 30 --campaign-name demo1
.venv/bin/python -m graphretr_opt.cli final --campaign-name demo1   # locked test, once
```

Stage-1 measured: seed recall@20 **0.1688** → one-shot rewrite **0.1837** on the
frozen 200-query gate (dense-retriever ceiling ref ~0.381). The architecture is
the Stage-1 subset of the full design; `slow_loop`/`scheduler`/`momentum`/
`agents/team`/`gepa_adapter`/`backends/neo4j` exist as documented seams that
Stage 2 turns on without moving the directory.

### Re-running the ETL
- It is **idempotent**: if `prime` already has the full 129,375 nodes /
  8,100,498 edges it exits immediately.
- To force a full rebuild: `docker exec graphretr-falkordb redis-cli GRAPH.DELETE prime`
  then re-run `etl_prime.py` (embeddings reload from `emb_cache.npy`, no
  re-embedding).
- To re-embed from scratch too: also delete `emb_cache.npy` / `emb_cache.meta.json`.

---

## Deviations from the briefing

1. **No passwordless sudo / only Python 3.12 system-wide** → used `uv` to
   provision a standalone **Python 3.11.15** venv (no apt needed). Docker was
   already installed and usable without sudo.
2. **Ports:** FalkorDB on **6380/3001** (not 6379/3000) — 6379 is taken by a
   pre-existing FalkorDB + Coolify stack on this shared host.
3. **Paths:** project under `/home/developer/ETH Agentic System Lab/Demo 1/graphretr-demo`
   (user is `developer`, not `root`); MLflow sqlite/artifacts rooted there.
   A compat symlink that briefly existed at `/home/developer/graphretr-demo`
   was removed 2026-06-10 — this directory is the single canonical location.
4. **MLflow** runs via `nohup` (no root for a systemd unit).
5. **CPU torch:** the default wheel pulled CUDA 13 libs (~2.7 GB) — replaced
   with `torch==2.12.0+cpu` since the box has no GPU.
6. **QA API:** `qa[i]` returns a 4-tuple `(query, q_id, answer_ids, meta_info)`
   — not the attribute-style `qa[i].query` shown in the briefing pseudocode.
7. **`max_seq_length=128`** on the embedder to keep the one-time CPU embed
   within a sane wall-clock.
8. **numpy 2.4.6** kept (rdkit prints a 1.x/2.x warning on import, but it is
   not on the prime code path and the full prime load/embed/query pipeline
   works end-to-end).

9. **Layout:** the optimizer lives under `src/graphretr_opt/` as a curated
   modular package (one responsibility per module, Stage-2 seams as docstring'd
   stubs) rather than a few large files — so the pipeline is navigable and the
   Stage-1→Stage-2 seam is just wiring. The old flat `optimizer/`+`scripts/`+
   `programs/` layout was replaced 2026-06-10; `runs/` holds campaign artifacts.
