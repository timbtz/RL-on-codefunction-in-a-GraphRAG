# Co-optimize ingestion+search — shared interface CONTRACT

This file pins the shared interface decisions so every piece (schema, ingestion,
search, reward adapter, attribution, tests) stays consistent. Authored under the
"author now, defer runtime" constraint (run15 is live; no Neo4j/node_modules in
this environment — nothing here has been executed/typechecked yet).

Worktree: `/home/developer/ETH Agentic System Lab/cooptimize-wt` (branch
`cooptimize-ingest-search`, off `run10c-openrouter`). NEVER edit the live tree
at `Demo 1/` — run15's subprocesses read from it.

## Graph DB
- **Neo4j 5 over bolt** (NOT FalkorDB — that's the STaRK target). Mirrors
  graphsearch's `langchain_neo4j.Neo4jGraph`.
- Env vars (graphmod side): `NEO4J_URI` (`bolt://localhost:7687`), `NEO4J_USER`
  (`neo4j`), `NEO4J_PASS` (`password`), `NEO4J_DB` (`neo4j`). Per-candidate graph
  isolation uses a distinct **database name** `graph_<ingest_hash>` (Neo4j 5
  multi-db) OR a fresh DB wipe keyed by hash — adapter decides; default: pass
  `NEO4J_DB=graph_<hash>`.

## Canonical node ids (MERGE keys — re-ingest is idempotent)
- `Person`    : `person:<slug(name|handle)>`        props `{id, name, info?}`
- `Component` : `component:<slug>`                    props `{id, name, info?}`
- `Entity`    : `entity:<slug>`  (generic/LLM/gazetteer) props `{id, name?}`
- `Chunk`     : `<ownerId>#c<index>`                  props `{id, content, source?}`
- `Document`  : `doc:<filename-stem>`                 props `{id, title, path}`
- `Message`   : `msg:<channel>:<seq>`                 props `{id, text, ts?, channel?}`
- `Ticket`    : `ticket:<KEY>` (e.g. `ticket:PROJ-12`) props `{id, key, title, description, status?}`

`slug(x)` = lowercase, non-alnum → `-`, collapse repeats, trim. Back canonical
ids with uniqueness constraints: `CREATE CONSTRAINT ... REQUIRE n.id IS UNIQUE`.

## Edges (relationship type names — used verbatim in Cypher + attribution)
Doc:    `HAS_CHUNK`(Document→Chunk,{index}) · `MENTIONS`(Chunk→Entity) ·
        `AUTHORED_BY`(Document→Person) · `ABOUT`(Document→Component) ·
        `REFERENCES`(Document→Document)
Chat:   `HAS_CHUNK`(Message→Chunk,{index}) · `SENT_BY`(Message→Person) ·
        `MENTIONS`(Chunk→Entity) · `REFERENCES`(Message→Document) ·
        `REPLY_TO`(Message→Message)
Jira:   `HAS_CHUNK`(Ticket→Chunk,{index}) · `ASSIGNED`(Ticket→Person) ·
        `REPORTED`(Ticket→Person) · `ABOUT`(Ticket→Component) ·
        `BLOCKS`(Ticket→Ticket) · `REFERENCES`(Ticket→Document) ·
        `MENTIONS`(Chunk→Entity)

The **LLM-extraction lever** later adds typed `Entity:<type>` labels and an
LLM-extracted `Entity→Entity` relation (e.g. `REVIEWED`) with `{sources:string[]}`
— per the commented hints in `textFile.module.ts`.

## Fulltext index (seed retrieval, no embeddings)
`CREATE FULLTEXT INDEX corpus_ft IF NOT EXISTS FOR (n:Chunk) ON EACH [n.content]`
Seed query: `CALL db.index.fulltext.queryNodes('corpus_ft', $q) YIELD node, score`.
Index name constant: **`corpus_ft`**.

## Citations / gold (retrieval_hit)
- Search returns a context string that includes citation tokens of the form
  **`[doc:<DocumentId>]`** for every Document it surfaces (and `[msg:...]` /
  `[ticket:...]` when the answer-bearing node is a message/ticket).
- A question's `gold` is the **node id** that proves the answer (a `doc:`/`msg:`/
  `ticket:` id). `retrieval_hit` = gold id appears (case-insensitive substring)
  in the returned context — mirrors `McqReward._retrieval_hit`.

## ingest_cost.json (TS writes, Python reads)
Written by `graphmod/src/ingestion/llm.ts` meter to
`graphmod/.ingest_cost/<ingest_hash>.json` (and a stable `ingest_cost.json` for
the last run). Schema:
```json
{ "ingest_hash": "abc123", "tokens": 0, "usd": 0.0,
  "calls": [{"field":"chat.msg:eng:7","tokens":123,"usd":0.0007}] }
```
Zero-LLM seed → `{tokens:0, usd:0.0, calls:[]}`.

## Chunk "card" requirement (CRITICAL — judge-free answering)
The MCQ answerer reads the **chunk text** the search returns, NOT the graph edges.
So ingestion must give every Document/Message/Ticket a primary Chunk whose
`content` is a readable serialization that NAMES related entities, e.g.:
- Ticket PROJ-12 chunk: `"Ticket PROJ-12: Migrate auth to OAuth 2.1. Status In
  Progress. Assignee Tim. Reporter Lasse. Components auth-redesign. References
  auth-redesign."`
- Document chunk: `"Document Auth Redesign Spec (doc:auth-redesign). Author Tim.
  Components auth-redesign."` + the body text split into further chunks.
- Message chunk: `"Message from Lasse: @tim the auth-redesign looks good - approving it."`
This way traversal gathers the right chunks and the answerer can actually read the
answer (a person name) out of context. Person/Component NAMES must appear in the
serialized chunk text of the nodes that point at them.

## ingest.ts CLI
`npx tsx graphmod/src/ingestion/ingest.ts <corpusDir> [--db graph_<hash>] [--hash <h>]`
- Reads `graphsearch/data/corpus/{docs/*.md, chat.json, tickets.json}`.
- Builds the graph (MERGE-upsert), creates constraints + `corpus_ft`.
- Writes `ingest_cost.json`. Exits non-zero on any build/schema error (so the
  reward adapter turns it into `crashed_frac` and rejects the candidate).
- Emits an `IngestReport` JSON to stdout: `{nodes, rels, byLabel, costPath}`.

## schema.json (export-schema.ts → consumed by extractor + attribution probe)
`npx tsx graphmod/scripts/export-schema.ts > graphmod/schema.json`
Shape: `{ labels: string[], relationships: [{type, from, to}], nodeProps: {...} }`
union over all Corpus modules. Used to (a) constrain the LLM extractor to allowed
labels/relationship triplets, (b) let `graph_inspect` summarize the schema.

## Reward adapter (Python) — the loop seam
`IngestSearchRewardAdapter.score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None)`
→ `MetricVector` (or `(MetricVector, rows)`). `fn` is the FileSet (overlay holds
`ingestion/extract.ts` + `search/search.py`). Mirrors `McqRewardAdapter` +
`McqReward`. Two-phase: hash the ingestion `.ts` overlay files → if graph_<hash>
not cached, materialize FileSet + `tsx ingest.ts` → run `search.py` over the MCQ
exam → score `mcq_accuracy`/`retrieval_hit` → read `ingest_cost.json` → emit
MetricVector with new ingest axes. Search-only edits reuse the cached graph.

Rows (return_rows) shape — reflect-compatible:
`{query, metrics:{recall@20,hit@1,hit@5,mrr}, retrieved, answer_ids, gold_ranks,
  recall@100, error, attribution}` where `attribution` ∈
`{NOT_INGESTED, ORPHANED, UNREACHABLE, RANKING}` (set by the graph probe).

## MetricVector new axes (objectives.py)
Add fields `ingest_usd: float = 0.0`, `ingest_tokens: int = 0` (one-time, per
built graph). Selection uses amortized
`total_usd_per_query = ingest_usd / max(1, n_queries) + usd_cost`.
Pareto tuple (pareto.py / qa_objectives.mcq_objective_tuple) gains `-total_usd_per_query`
in place of (or alongside) `-usd_cost`.

## Attribution buckets (fast_loop / graph_inspect)
For each missed query, probe the built graph read-only:
- `NOT_INGESTED` — gold node id absent from graph.
- `ORPHANED`     — gold node present but no path from any fulltext seed.
- `UNREACHABLE`  — path exists but search didn't reach it (depth/limit) → search fix.
- `RANKING`      — retrieved but ranked out → search fix.

## Read-only inspect gate (graph_inspect.py)
Reuse the regex firewall pattern from
`starksearch/src/stark_harness/qa_runner.py` (`_assert_safe`, `ReadOnlyGraphClient`):
reject `create|merge|set|delete|remove|drop|detach` + write procs, force a LIMIT,
truncate rows, scope to the current candidate's `graph_<hash>` DB, never expose MCQ gold.

## File inventory (all under the worktree)
graphmod/src/modules/Corpus/{Component,Entity,Chunk,Doc,Message,Ticket}/*.module.ts
graphmod/src/modules/Corpus/index.ts            (module list barrel for export-schema)
graphmod/src/ingestion/{ingest,extract,resolve,llm}.ts
graphmod/src/ingestion/loaders/{markdown,chat,jira}.ts
graphmod/scripts/export-schema.ts
graphsearch/data/corpus/{docs/*.md, chat.json, tickets.json}
graphsearch/data/dataset.json                    (~30-60 MCQs)
graphsearch/src/search/search.py
graphretr-demo/src/graphretr_opt/reward/ingest_search.py
graphretr-demo/src/graphretr_opt/optimizer/graph_inspect.py
graphretr-demo/src/graphretr_opt/config.py        (add ingest_search target)
graphretr-demo/src/graphretr_opt/reward/objectives.py + pareto.py (cost axes)
graphretr-demo/src/graphretr_opt/optimizer/fast_loop.py (attribution)
+ tests (graphmod tsx + python)
