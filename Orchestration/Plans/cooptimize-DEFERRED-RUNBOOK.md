# Co-optimize ingestion+search — DEFERRED runtime validation runbook

Everything in this plan was AUTHORED in the worktree `cooptimize-wt` (branch
`cooptimize-ingest-search`) under the "author now, defer runtime" constraint: the
box has ~1 GB free RAM while `run15_archipelago` is live, and standing up Neo4j +
running ingest/integration would risk OOM-killing run15. **Run NONE of the below
until run15 has finished and RAM is free.** Check first:

```bash
pgrep -af "graphretr_opt.cli (optimize|archipelago)"   # must be empty
free -m                                                 # want >3 GB available
```

## Why deferred (resource facts at authoring time)
- 4 CPUs, 7.5 GB RAM, ~1 GB free (run15 used most of it).
- No `node_modules` / `tsc` / `tsx` installed anywhere — graphmod TS never built here.
- FalkorDB (6379/6380) is run15/STaRK's DB; **Neo4j (7474/7687) is FREE** — no port clash.

## Order of operations (each gates the next)

### 0. Isolation note
Validate FROM THE WORKTREE so the live tree stays clean:
`cd "/home/developer/ETH Agentic System Lab/cooptimize-wt"`. The worktree shares
`.git` but has NO node_modules/.venv (gitignored) — install below.

### 1. Level 1 — TS install + typecheck (graphmod)
```bash
cd graphmod
npm install            # neo4j-driver + tsx + typescript + @types/node (devDeps)
npm run typecheck      # tsc --noEmit  -> EXPECT the FIRST real errors here
npx tsx scripts/export-schema.ts && test -f schema.json && echo OK
```
LIKELY FIRST FAILURES (unvalidated TS): defineModule's heavy generics across the
include-merged Corpus modules (Doc/Message/Ticket include Person/Component/Entity/
Chunk + Document). If `tsc` complains about relationship from/to keys, check the
include ALIASES equal the referenced node names (we aliased `Document: DocModule`).
tsx RUNS regardless of tsc (it strips types), so ingest can be smoke-tested even if
typecheck has residual complaints — but fix them for cleanliness.

### 2. Level 1 — Python lint
```bash
cd ../graphretr-demo && python -m pyflakes src/graphretr_opt   # or the project linter
```

### 3. Stand up Neo4j (NOT FalkorDB)
Neo4j 5, ports 7474/7687. **Neo4j Community is SINGLE-DB** (no named graph_<hash>
DBs) — the adapter's cache is wipe-and-rebuild of the default `neo4j` db keyed by a
loaded-hash marker. (Enterprise/multi-db is the future isolation; see adapter TODO.)
```bash
docker run -d --name cooptimize-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_max__size=512m \
  -e NEO4J_server_memory_pagecache_size=256m \
  neo4j:5
curl -sf localhost:7474 >/dev/null && echo OK
```
Env for both sides (graphmod getSession + graphsearch Neo4jGraph):
`NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASS=password NEO4J_DB=neo4j`
and `GRAPHSEARCH_NEO4J_URL=bolt://localhost:7687 GRAPHSEARCH_NEO4J_USER=neo4j
GRAPHSEARCH_NEO4J_PASSWORD=password GRAPHSEARCH_NEO4J_DATABASE=neo4j`.

### 4. Level 3 — seed ingest (zero-LLM)
```bash
cd graphmod
npx tsx src/ingestion/ingest.ts ../graphsearch/data/corpus
# expect: IngestReport JSON on stdout, ingest_cost.json = {tokens:0, usd:0, calls:[]}
# Cypher sanity:
#   MATCH (n) RETURN count(n)                         > expected (~ people4+comp3+docs5+chunks+msgs12+tickets5)
#   MATCH (:Ticket{id:'ticket:PROJ-12'})-[:REFERENCES]->(d)-[:AUTHORED_BY]->(p) RETURN p   -> person:tim
# re-run ingest -> identical counts (idempotency).
```

### 5. Level 2 — unit tests
```bash
cd graphmod && npm test          # tsx --test tests/Module/**  (loaders/resolve/extract pure; idempotency DB-gated)
cd ../graphretr-demo && PYTHONPATH=$PWD/src python -m tests.test_ingest_cost_amortized   # already passes
pytest tests/ -k "ingest or attribution or metric or inspect"
```

### 6. Hand-measure the seed baseline (Phase-1 exit)
Run search.py over ~5 MCQs manually against the built graph; confirm a believable
non-zero retrieval_hit/accuracy BEFORE wiring the optimizer. This de-risks the
TS<->Python seam (the plan's M0/M1 exit).

### 7. Level 3 — two-phase harness + optimize (M2 -> M3)
```bash
cd graphretr-demo
# reproduce the seed score THROUGH IngestSearchRewardAdapter first (M1 exit), then:
python -m graphretr_opt.cli optimize-ingest-search --steps 3   # add --llm-extraction for M4
```
- M2: keep `editable_files`/`ingest_editable_files` = search.py only (sanity on fixed graph).
- M3: add extract.ts; expect >=1 accepted ingestion edit clearing a NOT_INGESTED/ORPHANED
  bucket; confirm ingest_usd/ingest_tokens appear in MLflow.

### 8. Level 4 — manual + M4 lever
- Neo4j Browser (localhost:7474): confirm a known multi-hop path; re-run a missed
  free-text question after enabling the LLM-extraction lever (`ingest_llm_extraction`
  / `opts.llmExtraction` per field) and confirm accuracy rises with ingest cost on
  the frontier (the narrated user-story run).
- Needs an LLM key (OPENROUTER_API_KEY/OPENAI_API_KEY) for the answerer + the lever.
- **TODO (M4 wiring gap):** the LLM lever's PRIMARY trigger is the optimizer EDITING
  `extract.ts` to enable extraction per field (this path works end-to-end). The
  GLOBAL convenience switch `--llm-extraction`/`cfg.ingest_llm_extraction` is parsed
  but NOT yet threaded into the ingest subprocess: to make it functional, (a) have
  `IngestSearchRewardAdapter` append `--llm-extraction` to its `tsx ingest.ts ...`
  command when `cfg.ingest_llm_extraction` is true, and (b) have `ingest.ts`'s CLI
  map that flag to `opts.llmExtraction`. Two ~1-line edits; deferred because they
  can't be validated without Neo4j + an LLM key.

## CALIBRATION (the real risk, per plan confidence 6/10)
The single biggest determinant is whether the corpus gives the optimizer real,
attributable headroom. After step 4-6, MEASURE per-question:
- Do the free_text MCQs (q021-024, q032-035) actually FAIL pre-lever? If their
  bearing chunk is retrieved and the answerer reads the prose directly, they pass
  without extraction → no headroom. Tune phrasing / seed keywords so the gold node
  is reachable ONLY via the LLM-extracted edge named in each `gold_path`.
- Confirm the structured/multihop MCQs are clearable by the rule-based seed (else the
  baseline is too low and attribution is noise).

## Packaging note (corpus is gitignored)
`.gitignore` excludes `graphsearch/data` (repo policy). The hand-authored corpus
+ exam exist ON DISK in the worktree (so deferred validation works), but are NOT
tracked. To include them when committing this branch, force-add:
```bash
git add -f graphsearch/data/corpus graphsearch/data/dataset.json
```
(Left the policy `.gitignore` unchanged on purpose — do not broaden it.)
The other new files (graphmod Corpus modules, ingestion/, scripts/, search.py,
ingest_search.py, graph_inspect.py, the optimizer edits, tests, these plans) are
tracked normally.

## Cleanup
`docker rm -f cooptimize-neo4j` ; `git worktree remove ../cooptimize-wt` (after merge/PR).
