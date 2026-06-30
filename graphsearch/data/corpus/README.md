# Corpus + exam (co-optimize ingestion+search)

Tiny fixed corpus over three sources, authored so gold answers need cross-source
multi-hop graph paths. Eval is serial/slow, so the corpus is deliberately small.

## Source formats (what the ingestion loaders parse)

### `docs/*.md` — markdown loader
YAML-ish frontmatter between `---` fences, then markdown body:
```
---
id: doc:<stem>          # canonical Document id; equals doc:<filename stem>
title: <string>
author: <person handle> # -> AUTHORED_BY (person:<handle>)
components: [<slug>, ...] # -> ABOUT (component:<slug>)
---
<body>                   # split into Chunk(s); [[wikilink]] -> REFERENCES doc:<wikilink>
```

### `chat.json` — chat loader (array)
```
{ "seq": <int>, "channel": <str>, "author": <handle>, "ts": <iso8601>,
  "text": <str>, "reply_to": <seq?>, "references": [<doc stem>?] }
```
- id = `msg:<channel>:<seq>`. `author` -> SENT_BY. `reply_to` -> REPLY_TO (same channel).
- `[[wikilink]]` in text (and explicit `references`) -> REFERENCES doc:<stem>.
- `@handle` -> MENTIONS person:<handle> (gazetteer). `PROJ-\d+` -> MENTIONS the ticket.
- Component names in the controlled vocabulary -> MENTIONS component:<slug> (gazetteer).

### `tickets.json` — jira loader (array)
```
{ "key": "PROJ-12", "title": <str>, "description": <str>, "status": <str>,
  "assignee": <handle>, "reporter": <handle>,
  "components": [<slug>...], "references": [<doc stem>...], "blocks": [<key>...] }
```
- id = `ticket:<KEY>`. assignee->ASSIGNED, reporter->REPORTED, components->ABOUT,
  references->REFERENCES doc, blocks->BLOCKS ticket.

## The graph this corpus should produce (rule-based seed)
People: tim, lasse, mara, noah. Components: auth-redesign, billing, search-index.

Docs: auth-redesign(tim), billing-overview(mara), search-index-design(lasse),
onboarding(noah), incident-2026-05(mara).
Doc REFERENCES: auth-redesign->search-index-design; search-index-design->auth-redesign;
onboarding->{billing-overview, auth-redesign}; billing-overview->onboarding.

Tickets: PROJ-12(assignee tim, reporter lasse, about auth-redesign, refs auth-redesign),
PROJ-42(mara/noah, billing, refs billing-overview), PROJ-7(lasse/tim, search-index,
refs search-index-design), PROJ-19(tim/mara, auth-redesign, refs incident-2026-05,
BLOCKS PROJ-7), PROJ-33(noah/lasse, refs onboarding).

## Exam (`../dataset.json`)
33 MCQs. Item schema is what `graphsearch/qa/qa_substrate.py` reads:
`{id, question, choices[], answer_idx, source, closedbook_correct}` (extra keys —
`type/hops/free_text_only/requires_llm_extraction/gold_path` — are documentation
and ignored by the substrate). `source` is the gold node id used for
`retrieval_hit` (case-insensitive substring match against the search context, which
emits citation tokens like `[doc:auth-redesign]`).

- **structured** (1-hop, ~17): answerable from one structured edge the rule-based
  ingestion creates. The seed baseline should largely clear these.
- **multihop** (2-3 hop, ~8): cross-source paths (jira->doc->person, doc->doc->person,
  ticket->blocks->ticket->reported). Stress the traversal + the structured edges.
- **free_text** (8, ~24%): the fact lives ONLY in prose (chat approvals, postmortem
  attributions). Rule-based ingestion stores the bearing chunk but NOT the relation
  in `gold_path`. These are the headroom the **LLM-extraction lever** earns in M4;
  each `gold_path` names the edge the lever should materialize.

> CALIBRATION CAVEAT (deferred to runtime, run15-blocked): whether each free_text
> question is genuinely UNanswerable pre-lever (vs answerable because its bearing
> chunk is retrieved and the MCQ answerer reads the prose directly) must be measured
> through `IngestSearchRewardAdapter` once Neo4j is up. Tune phrasing / seed keywords
> then so the lever has real, attributable headroom. ~⅓ free-text is the target.
