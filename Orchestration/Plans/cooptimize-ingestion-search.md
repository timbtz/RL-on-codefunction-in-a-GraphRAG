# Feature: Co-optimize graph **ingestion + search** (with optimizer-controlled LLM extraction & full-pipeline cost metering)

The following plan should be complete, but it is important to validate documentation and codebase
patterns and task sanity before implementing. Pay special attention to the naming of existing
utils/types/models (`MetricVector`, `FileSet`, `defineModule`, `assertValid`, `McqRewardAdapter`) and
import from the right files.

## Feature Description

Extend the EvoRetrieve optimizer so a single candidate spans **both** the graph *ingestion* code
(TypeScript, on `graphmod`) **and** the *search* code (Python), evaluated by one two-phase rollout:
`tsx ingest.ts → Neo4j → python search → judge-free MCQ exam`. The optimizer evolves the schema, the
ingestion rules, and the search together; it may **selectively insert LLM extraction** at chosen
points in ingestion (e.g. pull entities out of a long, unformatted comment that is otherwise only
stored as a `Chunk`), and it is **priced on the total LLM token cost of the whole ingest+search
pipeline**, not just search. This is the README's stated Phase-2 / Roadmap item; ~80% of the
machinery (multi-file `FileSet`, Pareto pool, USD metering, read-only Cypher gate, Neo4j+MCQ harness)
already exists and is reused.

## User Story

> **As** the EvoRetrieve optimizer (LLM mutation operator),
> **I want** to start from a **zero-LLM, rule-based** ingestion + search candidate, run the exam, and
> for every unanswered question receive feedback that tells me **whether the miss is a search failure
> or an ingestion failure** — with a **read-only tool to query the live Neo4j graph** to confirm
> whether the needed node / entity / relationship exists and *where* — so that, per failure, I can
> decide to **(a) tweak the search**, **(b) change the schema**, or **(c) change the ingestion**
> (including **adding an LLM extractor** that lifts entities/relations out of long unformatted text
> into nodes/edges defined by the TypeScript module),
> **so that** retrieval accuracy rises while a **cost function that meters the total LLM tokens of the
> whole ingest+search process** stops me from spending more than the retrieval gain is worth.

**The intended trajectory the system should demonstrate (the demo narrative):**
1. **Candidate 0** ingests with **no LLM** (rule-based) and searches. Exam runs → say 0.45.
2. Feedback: "Questions Q7, Q12, Q19 missed." For each, the failure is **attributed** (search vs
   ingestion) and the optimizer **queries Neo4j read-only** to check the graph it built.
3. It finds Q7's answer node *exists but is unreachable* → **search fix** (extend traversal).
4. It finds Q12's required relationship *was never created* (the fact lived in a free-text comment that
   rule-based extraction only stored as a `Chunk`) → **ingestion fix**: add an **LLM extractor** for
   that comment field, materializing `Entity` nodes + edges per the `defineModule` schema.
5. It re-runs. Accuracy rises; the new LLM ingestion call shows up as **ingest token cost** on the
   Pareto frontier — accepted only if the accuracy gain justifies the spend.

## Problem Statement

Today a rollout searches a **fixed, pre-built** graph (`fast_loop.py`); ingestion is out of scope and
`graphmod` is "not yet under optimizer control" (top-level `README.md`). Therefore the optimizer
cannot fix the single largest source of graph-RAG retrieval failures: **the graph was built wrong**
(entity not resolved, relationship never extracted, fact buried in prose). It also has **no way to
trade the cost of building a richer graph (LLM extraction) against the retrieval value it produces**,
and **no feedback that distinguishes "search didn't reach it" from "it was never ingested."**

## Solution Statement

1. Make the **candidate `FileSet` span `ingest.ts` (TS) + `search.py` (Python)** by widening the
   `editable_files` whitelist; the loop is already multi-file.
2. Add a **two-phase reward adapter** that (re)builds a per-hash Neo4j graph via `tsx ingest.ts`, then
   scores search over the MCQ exam — reusing `McqRewardAdapter`'s judge-free scoring.
3. Add **ingestion-vs-search failure attribution** + a **read-only `inspect_graph` tool** (reusing the
   existing STARK read-only Cypher gate) so the optimizer can diagnose the graph it built.
4. Make **LLM extraction an optimizer-controllable lever inside `ingest.ts`** — opt-in per
   source/field, governed by the `graphmod` schema, every call metered.
5. Extend the **cost function to meter total ingest+search LLM tokens/USD** and select on the
   accuracy-vs-**total**-cost Pareto frontier (ingest cost is amortized across the query set).

## Feature Metadata

**Feature Type**: New Capability (extends an existing optimizer)
**Estimated Complexity**: High (cross-language harness + new metrics + new feedback channel)
**Primary Systems Affected**: `graphretr-demo` (optimizer loop, config, reward, reflection, agents),
`graphmod` (schema modules + new ingestion layer), `graphsearch` (corpus, exam, search target), Neo4j.
**Dependencies**: `neo4j-driver` (TS, already in `graphmod`), Neo4j 5.x via Docker, `tsx`, Python 3.11
optimizer stack, an LLM backend for extraction (OpenRouter, the same metering path search already uses).

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: READ THESE BEFORE IMPLEMENTING
> Line numbers are approximate (from exploration) — verify on open.

- `README.md` (lines 160–207) — Phase-2 roadmap + the bloat/crash-spiral warning that drives the
  tight-scope, tight-complexity-gate discipline here.
- `graphretr-demo/src/graphretr_opt/optimizer/fast_loop.py`
  - control loop (~669–1030) — Why: rollout → reflect → propose → gate → accept; where the two-phase
    rollout and `inspect_graph` hook in.
  - `_reflect` (~89–148) and `_failure_record` (~150–189) — Why: the `missed`/`misranked` bucketing to
    EXTEND with `NOT_INGESTED`/`ORPHANED`/`UNREACHABLE`/`RANKING`.
- `graphretr-demo/src/graphretr_opt/config.py` (~138–148, `editable_files` / `stark_editable_files`)
  — Why: add the `IngestSearch` target whitelist (`ingest.ts`, `search.py`).
- `graphretr-demo/src/graphretr_opt/artifact/file_set.py` — Why: candidate overlay; confirm it treats
  files as opaque text (so a `.ts` file in the overlay is fine).
- `graphretr-demo/src/graphretr_opt/reward/objectives.py` — Why: `MetricVector` + Pareto dominance;
  add `ingest_usd` / `ingest_tokens` axes and the amortized `total_usd_per_query`.
- `graphsearch/reward/__init__.py` (`McqRewardAdapter`) and `graphsearch/reward/qa_objectives.py`
  (`mcq_accuracy`, `retrieval_hit`) — Why: MIRROR this to build `IngestSearchRewardAdapter`.
- `graphsearch/qa/qa_substrate.py` — Why: train/gate/meta/promote splits to reuse for our exam.
- `graphsearch/src/common/service/search/agentic_graph_traversal_search_service.py` — Why: the existing
  editable Neo4j search target; the seed `search.py` MIRRORS its fulltext-seed → traversal shape.
- STARK **read-only Cypher gate** in `starksearch/src/stark_search/` (AST gate: no `CREATE/MERGE/SET`,
  forced `LIMIT`) — Why: REUSE verbatim as the safety layer for `inspect_graph`. **Verify exact path.**
- `graphmod/src/core/module.ts` (`defineModule` 90–170; `Module` interface 25–70, incl. `assertValid`
  159–161, `fromLazyGraph` 144–146) — Why: the schema factory + the in-tx validation safety rail.
- `graphmod/src/core/types.ts` (`PropType`, `NodeDef`, `RelationshipDef`, `Cardinality`,
  `ModuleSchema`) — Why: the schema vocabulary the optimizer edits and the ingestor maps onto.
- `graphmod/src/modules/Document/TextFile/textFile.module.ts` (nodes 10–34, rels 37–60) — Why: the
  canonical schema example, **with commented-out LLM-extraction hints** (`Entity: ["Entity",
  "<entity_type>"]`, "relation extracted by LLM, e.g. isAuthorOf") — this is the seam the LLM-extraction
  lever turns on.
- `graphmod/src/modules/Person/person.service.ts` (`create` 39–69) — Why: the write pattern
  (`tx.run` → `assertValid` same tx). **GOTCHA**: uses `CREATE` + `crypto.randomUUID()` → NOT
  idempotent; ingestion must MERGE on canonical ids (see Patterns).
- `graphmod/examples/session.ts` — Why: `getSession()` via `neo4j-driver` (env: `NEO4J_URI/USER/PASS/DB`).
- `graphmod/examples/meeting-test.ts` — Why: end-to-end service-call + `fromLazyGraph` usage to mirror.

### New Files to Create
- `graphmod/src/ingestion/ingest.ts` — the deep ingestion module: `ingest(session, corpusDir) →
  IngestReport`. **Optimizer-editable.**
- `graphmod/src/ingestion/loaders/{markdown,chat,jira}.ts` — source adapters → `RawRecord`.
- `graphmod/src/ingestion/extract.ts` — rule-based extractor + **optimizer-insertable LLM-extraction
  calls** (metered); writes via MERGE-upsert. **Optimizer-editable.**
- `graphmod/src/ingestion/resolve.ts` — entity resolver (canonical-id MERGE).
- `graphmod/src/ingestion/llm.ts` — thin metered LLM client returning `{json, tokens, usd}` written to
  a sidecar `ingest_cost.json` the Python adapter reads.
- `graphmod/scripts/export-schema.ts` — dumps `Module.schema` → `schema.json` (so the extractor + the
  attribution probe know allowed labels/relationships).
- `graphsearch/data/corpus/{docs/*.md, chat.json, tickets.json}` — the tiny fixed corpus.
- `graphsearch/data/dataset.json` (extend) — ~30–60 MCQs over cross-source multi-hop paths.
- `graphretr-demo/src/graphretr_opt/reward/ingest_search.py` — `IngestSearchRewardAdapter`
  (two-phase, per-hash graph cache, reads `ingest_cost.json`).
- `graphretr-demo/src/graphretr_opt/optimizer/graph_inspect.py` — `inspect_graph(cypher)` + canned
  helpers behind the read-only AST gate.
- `graphsearch/src/search/search.py` — the editable search target (seed = fulltext seed → bounded
  traversal), MIRRORS the existing agentic traversal service.

### Relevant Documentation — READ BEFORE IMPLEMENTING
- [Neo4j GraphRAG KG Builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)
  — pipeline order, entity resolvers, graph pruner. Why: the ingestion shape we mirror in TS.
- [Neo4j full-text indexes (Cypher Manual)](https://www.neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/)
  — `CREATE FULLTEXT INDEX`, `db.index.fulltext.queryNodes`, Lucene syntax/score. Why: no-embedding seed retrieval.
- [Variable-length / quantified paths](https://neo4j.com/docs/cypher-manual/current/patterns/variable-length-paths/)
  — bounded `*1..n` traversal. Why: multi-hop search; ALWAYS bound the hop count.
- [neo4j-driver (npm)](https://www.npmjs.com/package/neo4j-driver) + [TS guide](https://dev.to/adamcowley/using-typescript-with-neo4j-478c)
  — Why: the TS ingestion runtime; parameterize all Cypher.
- [LLM entity/relation extractor](https://neo4j.com/docs/neo4j-graphrag-python/current/_modules/neo4j_graphrag/experimental/components/entity_relation_extractor.html)
  — Why: the schema-constrained extraction prompt shape for the LLM-extraction lever.

### Patterns to Follow

**Schema definition (graphmod, `textFile.module.ts`):** declare node `labels` + `properties` and
typed `relationships` with `cardinality`; the optimizer edits this object to add labels/edges.

**Idempotent write (REPLACES the `CREATE`+UUID pattern in `person.service.ts`):**
```ts
// canonical id, not crypto.randomUUID() — so re-ingest is a no-op, not a duplicate
await tx.run(
  `MERGE (p:Person { id: $id })
   ON CREATE SET p.name = $name
   ON MATCH  SET p.name = coalesce(p.name, $name)`,
  { id: canonicalPersonId(name), name });
await PersonModule.assertValid(tx, id);   // in-tx schema check → rolls back bad edits
```
Back every canonical id with a uniqueness constraint (`CREATE CONSTRAINT ... REQUIRE e.id IS UNIQUE`).

**Metered LLM extraction (the optimizer-insertable lever, `extract.ts` + `llm.ts`):**
```ts
// rule-based ALWAYS stores the raw text as a Chunk (cheap, deterministic):
const chunkId = await upsertChunk(tx, record);            // 0 tokens
// the optimizer may ADD this block for a chosen field/source — every call metered:
const { entities, relations, tokens, usd } = await extractWithLLM(record.text, schemaJson);
meter.add(tokens, usd);                                   // → ingest_cost.json
for (const e of entities) await upsertEntity(tx, e);      // entities created FROM the comment
for (const r of relations) await upsertRelation(tx, r);   // per defineModule schema
```

**Read-only inspection (Python, behind the STARK AST gate):**
```python
inspect_graph("MATCH (p:Person)-[:OWNED_BY]-(d:Document) RETURN p.id, d.id LIMIT 25")
# helpers: find_node(name) · neighbors(id, depth) · count_by_label() · schema_summary()
# gate rejects CREATE/MERGE/SET/DELETE; forces LIMIT; times out. View-only.
```

**Cost vocabulary (extend `MetricVector` in `objectives.py`):** add `ingest_usd`, `ingest_tokens`
(one-time, per built graph) alongside existing per-query `usd_cost`. Selection uses
`total_usd_per_query = ingest_usd / N_queries + search_usd_per_query` so an expensive LLM ingestion is
**amortized** across the exam — worth it if it lifts accuracy broadly, wasteful if it helps one query.

**Anti-patterns (from `README.md` Phase-2 failure):** do NOT widen editable files beyond 2–3 early;
keep the complexity/crash gate tight; keep the corpus small (eval is serial/slow).

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation (M0) — schema, corpus, exam, zero-LLM seed
Stand up Neo4j; author the `graphmod` schema modules for docs/chat/jira; hand-write the **rule-based**
seed `ingest.ts` and seed `search.py`; author the corpus + ~30 MCQs designed so answers require
cross-source multi-hop links. **Exit:** graph opens in Neo4j Browser; seed scores a believable
non-zero baseline by hand.

### Phase 2: Core Implementation (M1) — the two-phase eval harness (no mutation)
Build `IngestSearchRewardAdapter`: `tsx ingest.ts → Neo4j(graph_<hash>) → python search → MCQ score`,
with **graph caching keyed by the hash of the ingestion `.ts` files** (search-only edits reuse the
graph) and ingest cost read from `ingest_cost.json`. **Exit:** reproduce the seed number through the
harness; the riskiest plumbing (TS↔Python seam) is hardened before any LLM edits.

### Phase 3: Integration (M2 → M3) — bring ingestion + feedback + cost under the optimizer
M2: widen `editable_files` to `search.py` only; run a few steps on the fixed graph (sanity).
M3: add `ingest.ts` to scope; add the `NOT_INGESTED/ORPHANED/UNREACHABLE/RANKING` attribution; add
`ingest_usd`/`ingest_tokens` to `MetricVector` + amortized Pareto. **Exit:** ≥1 accepted **ingestion**
edit that raises `retrieval_hit` by clearing a `NOT_INGESTED`/`ORPHANED` bucket.

### Phase 4: LLM-extraction lever + inspection tool + validation (M4)
Wire `inspect_graph` into reflection/propose; enable the **metered LLM-extraction lever** in
`extract.ts`; optionally let the optimizer edit the `defineModule` schema. **Exit:** a narrated run
where the optimizer queries the graph, sees a fact was never extracted, **adds an LLM extractor** for
that field, accuracy rises, and the ingest token cost appears on the frontier.

---

## STEP-BY-STEP TASKS
IMPORTANT: execute in order; each task is atomic and independently testable.

### CREATE infra: Neo4j + schema export
- **IMPLEMENT**: docker-compose with `neo4j:5` (ports 7474/7687, `NEO4J_AUTH`); `export-schema.ts`.
- **PATTERN**: `graphmod/examples/session.ts` for connection env vars.
- **VALIDATE**: `docker compose up -d && curl -s localhost:7474 >/dev/null && echo OK`
- **VALIDATE**: `cd graphmod && npx tsx scripts/export-schema.ts && test -f schema.json`

### CREATE `graphmod/src/modules/**` schema for docs/chat/jira
- **IMPLEMENT**: `Document`/`Chunk`/`Entity` (mirror TextFile), `Message`(chat) with `SENT_BY`,
  `MENTIONS`, `REFERENCES`, `REPLY_TO`; `Ticket`(jira) with `ASSIGNED`,`REPORTED`,`ABOUT`(Component),
  `BLOCKS`,`REFERENCES`. Shared `Person`/`Component` entity layer across all three.
- **PATTERN**: `textFile.module.ts` (nodes 10–34, rels 37–60).
- **GOTCHA**: keep the same `Entity` node generic now; the LLM lever later adds `<entity_type>` labels.
- **VALIDATE**: `cd graphmod && npm run typecheck`

### CREATE corpus + exam
- **IMPLEMENT**: `graphsearch/data/corpus/` (~5 md, ~12 chat msgs, ~5 tickets) authored so gold answers
  need cross-source paths (e.g. ticket→referenced doc→owner; chat→mentions→assignee). ~30–60 MCQs.
- **GOTCHA**: at least ~⅓ of questions must require a fact that lives **only in free-text** (so the
  LLM-extraction lever has something to earn in M4).
- **VALIDATE**: a `jq` schema check on `dataset.json`; manual: each MCQ has a known graph path.

### CREATE rule-based seed `ingest.ts` (loaders → extract → resolve → MERGE → fulltext index)
- **IMPLEMENT**: structured-field edges (assignee/author/components/links) + regex tokens
  (`PROJ-\d+`, `@handle`, `[[wikilink]]`, URLs) + **gazetteer** MENTIONS (controlled-vocabulary string
  match of known entity names/aliases in chunk text). MERGE on canonical ids; `CREATE FULLTEXT INDEX`.
- **PATTERN**: idempotent MERGE block above; `assertValid` after writes.
- **VALIDATE**: `cd graphmod && npx tsx src/ingestion/ingest.ts ../graphsearch/data/corpus`
  then a Cypher count: `MATCH (n) RETURN count(n)` > expected; a known path query returns a row.

### CREATE seed `search.py` (fulltext seed → bounded traversal)
- **PATTERN**: MIRROR `agentic_graph_traversal_search_service.py`.
- **GOTCHA**: bound traversal depth; return `[doc:ID]` citations for `retrieval_hit`.
- **VALIDATE**: run search over 5 MCQs manually → non-zero accuracy.

### CREATE `IngestSearchRewardAdapter` (M1)
- **IMPLEMENT**: build/reuse `graph_<ingest_hash>`; shell `tsx ingest.ts`; run search; score
  `mcq_accuracy`,`retrieval_hit`; read `ingest_cost.json`; emit `MetricVector`.
- **PATTERN**: MIRROR `graphsearch/reward/__init__.py` (`McqRewardAdapter`).
- **GOTCHA**: serialize ingestion; surface `tsx` non-zero exit / build error into `crashed_frac` +
  self-repair text (bad ingest edits must be rejected, not silently scored).
- **VALIDATE**: harness reproduces the hand-measured seed score within noise.

### UPDATE `config.py` — add `IngestSearch` target (M2→M3)
- **IMPLEMENT**: `editable_files = ("ingestion/extract.ts", "search/search.py")`; start with 2.
- **VALIDATE**: `python -m graphretr_opt.cli optimize --target ingest_search --steps 2` runs.

### UPDATE `fast_loop._reflect` / `_failure_record` — ingestion attribution (M3)
- **IMPLEMENT**: for each missed query, probe the built graph (read-only) → bucket
  `NOT_INGESTED` (gold node absent) / `ORPHANED` (no path from seed) / `UNREACHABLE` (path exists,
  search missed) / `RANKING`; render which side to fix.
- **PATTERN**: extend existing GENERATION/RANKING attribution (~150–189).
- **VALIDATE**: a fixture graph with a deliberately missing edge yields `ORPHANED` for the right query.

### UPDATE `objectives.py` — ingest cost axes + amortized Pareto (M3)
- **IMPLEMENT**: add `ingest_usd`,`ingest_tokens`; selection on `total_usd_per_query`.
- **VALIDATE**: unit test: a candidate with higher accuracy but huge ingest cost is dominated unless
  the per-query amortized gain clears the cost.

### CREATE `graph_inspect.py` + wire into agent (M4)
- **IMPLEMENT**: `inspect_graph(cypher)` + helpers behind the STARK read-only AST gate; expose to the
  mutator during reflect/propose; scope to the current candidate's graph; never expose MCQ gold.
- **PATTERN**: REUSE the STARK Cypher AST gate.
- **VALIDATE**: `inspect_graph("CREATE (x)")` is rejected; a read query returns rows with forced LIMIT.

### ADD metered LLM-extraction lever in `extract.ts` + `llm.ts` (M4)
- **IMPLEMENT**: opt-in per source/field; rule-based always stores the `Chunk`, LLM optionally adds
  `Entity`/relations constrained to `schema.json`; write `{tokens, usd}` to `ingest_cost.json`.
- **GOTCHA**: constrain the prompt to allowed labels/relationship triplets; dedupe against existing
  entities (feed the resolver's canonical list) to avoid duplicate nodes.
- **VALIDATE**: toggling the lever on a long-comment fixture creates the expected entity + edge AND
  increases `ingest_tokens`; toggling off reverts both.

---

## TESTING STRATEGY

### Unit Tests
- graphmod (`tsx --test`, mirror `tests/Module/**`): loaders → `RawRecord`; gazetteer match;
  canonical-id MERGE idempotency (ingest twice → identical node count); `assertValid` rolls back a
  schema-violating edit; LLM lever creates schema-valid entities (with a stubbed `llm.ts`).
- optimizer (python): attribution bucketing on fixture graphs; `MetricVector` amortized dominance;
  `inspect_graph` gate rejects writes / forces LIMIT.

### Integration Tests
- Full two-phase rollout on the corpus through `IngestSearchRewardAdapter` reproduces the seed score.
- A scripted 3–5 step optimize run produces ≥1 accepted ingestion edit and logs the cost axes.

### Edge Cases
- Re-ingest idempotency (no duplicates); empty/garbled source file (loader skips, logs warning);
  `tsx` build error → `crashed_frac`, candidate rejected; LLM returns off-schema entity → pruned;
  entity duplicated across sources → resolver collapses to one node; unbounded-traversal guard.

---

## VALIDATION COMMANDS
Execute every command to ensure zero regressions and feature correctness.

### Level 1 — Syntax & Style
- `cd graphmod && npm run typecheck`
- `cd graphretr-demo && python -m pyflakes src/graphretr_opt` (or project linter)

### Level 2 — Unit Tests
- `cd graphmod && npm test`
- `cd graphretr-demo && pytest tests/ -k "ingest or attribution or metric or inspect"`

### Level 3 — Integration
- `docker compose up -d`
- `cd graphmod && npx tsx src/ingestion/ingest.ts ../graphsearch/data/corpus`
- `cd graphretr-demo && python -m graphretr_opt.cli optimize --target ingest_search --steps 3`

### Level 4 — Manual Validation
- Open Neo4j Browser (localhost:7474); confirm a known multi-hop path exists; re-run a missed question
  after an accepted ingestion edit and confirm it now passes; check the run's cost axes in MLflow.

### Level 5 — Optional
- Held-out bake-off re-score of survivors on a disjoint exam split (reuse existing bake-off path).

---

## ACCEPTANCE CRITERIA
- [ ] One candidate spans `ingest.ts` + `search.py`; the two-phase rollout runs end-to-end.
- [ ] Seed (zero-LLM) ingestion scores a believable non-zero baseline through the harness.
- [ ] Re-ingestion is idempotent (no duplicate nodes); per-hash graph cache skips re-ingest on
      search-only edits.
- [ ] Failure feedback attributes misses to `NOT_INGESTED/ORPHANED/UNREACHABLE/RANKING`.
- [ ] `inspect_graph` is read-only (writes rejected, LIMIT forced) and usable by the optimizer.
- [ ] `MetricVector` meters `ingest_usd/ingest_tokens`; selection uses amortized `total_usd_per_query`.
- [ ] The LLM-extraction lever is optimizer-insertable, schema-constrained, and metered.
- [ ] ≥1 accepted ingestion edit raises `retrieval_hit`; ≥1 narrated run shows the full user-story
      trajectory (rule-based → query graph → add LLM extractor → accuracy↑ at justified cost).
- [ ] No regression to existing STARK / graphsearch targets; complexity/crash gate kept tight.

## COMPLETION CHECKLIST
- [ ] All tasks completed in order; each task's validation passed.
- [ ] Unit + integration suites pass; typecheck/lint clean.
- [ ] Manual Neo4j + MLflow validation confirms the trajectory and cost metering.
- [ ] Acceptance criteria met; plan reviewed for the README's anti-bloat discipline.

---

## NOTES

**Why this is the right application story.** It is the README's *stated* next step (Phase 1 = 0.44/0.28
held-out on STARK), not a pivot; it demonstrates the novel claim — **code-evolution where an ingestion
change and a search change are coupled and must be optimized together**, with a judge-free exam and
**total-pipeline USD** on the Pareto axis; and the attribution + `inspect_graph` make the optimizer's
reasoning legible. Even 2–3 accepted ingestion edits is a sufficient eye-catcher.

**On "rule-based vs LLM" extraction (design rationale).** The schema is never induced from data — it
is human-authored in `defineModule` and *evolved by the optimizer*. Rule-based extraction works for
the POC because the three sources carry most of their graph in **structure** (Jira/chat fields),
**typed tokens** (regex), and **gazetteer** matches against a controlled vocabulary. What rules
*cannot* do — implicit relations / novel entities in free prose — is precisely the gap the optimizer
closes by **adding the metered LLM extractor**. Worked example: a chat comment *"@tim the auth-redesign
looks good"* yields, rule-based, only `MENTIONS(tim)` + `REFERENCES(doc)`; if a question needs
`(tim)-[:REVIEWED]->(doc)`, the miss is attributed `NOT_INGESTED`, and the optimizer's fix is to turn
on LLM extraction for that comment field → the `REVIEWED` edge is created → accuracy rises → the token
cost is amortized across the exam on the frontier.

**Key trade-off decided:** ingestion runtime = **TypeScript / full graphmod** (cross-language `tsx →
Neo4j` seam hardened in M1); extraction = **rule-based first, LLM lever added under optimizer
control**; status = **plan-only**.

**Confidence (one-pass implementation): 6/10.** High-value but genuinely complex: the cross-language
harness, graph-isolation/caching, and corpus+exam design are the risk drivers. M0/M1 exist to retire
the biggest risk (the TS↔Python round-trip) before any LLM editing; the corpus design is the single
biggest determinant of whether the optimizer has real, attributable headroom to climb.
