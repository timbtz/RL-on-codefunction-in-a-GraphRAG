// ============================================================
// extract.ts — rule-based extractor (chunk cards, structured edges, regex +
//              gazetteer MENTIONS) + the DEFAULT-OFF, metered LLM lever.
// ============================================================
//
// This is the optimizer-editable file (the reward adapter overlays it). It holds
// (a) the MERGE-upsert writer helpers, (b) the chunk "card" serialization that
// NAMES related entities so the MCQ answerer can read answers out of chunk text,
// (c) the rule-based edge/MENTIONS extraction, and (d) `extractWithLLM` — the
// metered, schema-constrained lever that recovers prose-only relations.
//
// Cypher labels/relationship types are interpolated from a FIXED vocabulary
// (never user input); every value is a bound parameter.

import type { GraphTransaction } from "../core/types";
import {
  Resolver,
  REL_TARGET_LABEL,
  personId,
  entityId,
  chunkId,
  slug,
  type RawRecord,
  type RawRef,
  type RelType,
  type SourceKind,
} from "./resolve";
import type { LlmClient, Meter } from "./llm";

// ---- Regex tokens ----

export const TICKET_RE = /\bPROJ-\d+\b/g;
export const HANDLE_RE = /@([A-Za-z0-9_-]+)/g;
export const URL_RE = /https?:\/\/[^\s)\]]+/g;
export const WIKILINK_TOKEN_RE = /\[\[([^\]]+)\]\]/g;

// ============================================================
// Writer helpers (idempotent MERGE-upserts)
// ============================================================

/** MERGE a node on its canonical id, upserting props on both create and match. */
export async function mergeNode(
  tx: GraphTransaction,
  label: string,
  id: string,
  props: Record<string, unknown>,
): Promise<void> {
  await tx.run(
    `MERGE (n:${label} {id: $id})
     ON CREATE SET n += $props
     ON MATCH  SET n += $props`,
    { id, props },
  );
}

/**
 * MERGE a directed edge between two existing nodes (matched by id+label).
 * If either endpoint is absent the MATCH yields no rows and nothing is written
 * (safe for forward/dangling references). Optional relationship props are set.
 */
export async function mergeEdge(
  tx: GraphTransaction,
  fromLabel: string,
  fromId: string,
  rel: string,
  toLabel: string,
  toId: string,
  relProps?: Record<string, unknown>,
): Promise<void> {
  const setClause = relProps ? "SET r += $relProps" : "";
  await tx.run(
    `MATCH (a:${fromLabel} {id: $fromId}), (b:${toLabel} {id: $toId})
     MERGE (a)-[r:${rel}]->(b)
     ${setClause}`,
    { fromId, toId, relProps: relProps ?? {} },
  );
}

// ============================================================
// Chunk "card" serialization (CRITICAL — judge-free answering)
// ============================================================

interface ChunkSpec {
  id: string;
  index: number;
  content: string;
  source: SourceKind;
}

/** Group structured-ref display names by relationship type. */
function namesByRel(refs: RawRef[]): Map<RelType, string[]> {
  const m = new Map<RelType, string[]>();
  for (const r of refs) {
    if (!r.name) continue;
    const arr = m.get(r.rel) ?? [];
    if (!arr.includes(r.name)) arr.push(r.name);
    m.set(r.rel, arr);
  }
  return m;
}

function clause(label: string, names: string[] | undefined): string {
  return names && names.length ? ` ${label} ${names.join(", ")}.` : "";
}

/**
 * Build the primary "card" chunk (index 0) for a record. The card NAMES every
 * related person/component/doc/ticket so traversal-gathered context lets the
 * answerer read the answer (usually a person name) directly out of the text.
 * Documents (body) and Tickets (description) additionally split their prose
 * into further chunks (index 1..n).
 */
export function buildChunks(rec: RawRecord): ChunkSpec[] {
  const by = namesByRel(rec.refs);
  const chunks: ChunkSpec[] = [];

  let card: string;
  if (rec.label === "Document") {
    const title = String(rec.props.title ?? rec.id);
    card =
      `Document ${title} (${rec.id}).` +
      clause("Author", by.get("AUTHORED_BY")) +
      clause("Components", by.get("ABOUT")) +
      clause("References", by.get("REFERENCES"));
  } else if (rec.label === "Message") {
    const author = (by.get("SENT_BY") ?? [])[0] ?? "unknown";
    card =
      `Message from ${author}: ${String(rec.props.text ?? "")}` +
      clause("References", by.get("REFERENCES"));
  } else if (rec.label === "Meeting") {
    const title = String(rec.props.title ?? rec.id);
    const date = rec.props.date ? ` on ${rec.props.date}` : "";
    card =
      `Meeting ${title} (${rec.id})${date}.` +
      clause("Attendees", by.get("ATTENDED")) +
      clause("Discusses tickets", by.get("DISCUSSES")) +
      clause("Components", by.get("ABOUT")) +
      clause("References", by.get("REFERENCES"));
  } else if (rec.label === "PullRequest") {
    const num = String(rec.props.number ?? "");
    const title = String(rec.props.title ?? "");
    const desc = rec.props.description ? ` ${String(rec.props.description)}` : "";
    card =
      `Pull request #${num} ${title} (${rec.id}).${desc}` +
      clause("Author", by.get("AUTHORED_BY")) +
      clause("Reviewers", by.get("REVIEWED_BY")) +
      clause("Fixes tickets", by.get("FIXES")) +
      clause("Components", by.get("ABOUT")) +
      clause("References", by.get("REFERENCES"));
  } else {
    // Ticket
    const key = String(rec.props.key ?? rec.id);
    const title = String(rec.props.title ?? "");
    const status = rec.props.status ? ` Status ${rec.props.status}.` : "";
    card =
      `Ticket ${key}: ${title}.` +
      status +
      clause("Assignee", by.get("ASSIGNED")) +
      clause("Reporter", by.get("REPORTED")) +
      clause("Components", by.get("ABOUT")) +
      clause("References", by.get("REFERENCES")) +
      clause("Blocks", by.get("BLOCKS"));
  }

  chunks.push({ id: chunkId(rec.id, 0), index: 0, content: card.trim(), source: rec.kind });

  // Any record body (document, meeting transcript, ticket description) splits into further chunks (paragraphs / utterances).
  if (rec.body) {
    const paras = rec.body
      .split(/\r?\n\s*\r?\n/)
      .map((p) => p.replace(/\s+/g, " ").trim())
      .filter(Boolean);
    paras.forEach((p, i) => {
      const idx = i + 1;
      chunks.push({ id: chunkId(rec.id, idx), index: idx, content: p, source: rec.kind });
    });
  }

  return chunks;
}

// ============================================================
// MENTIONS targets (regex + gazetteer)
// ============================================================

interface MentionTarget {
  id: string;
  label: string;
  /** props to upsert when the node may not already exist (Person/Entity) */
  ensure?: Record<string, unknown>;
}

/** Collect distinct MENTIONS targets for a record from its prose. */
export function collectMentions(rec: RawRecord, resolver: Resolver): MentionTarget[] {
  const byId = new Map<string, MentionTarget>();
  const text = rec.text;

  // Gazetteer (known person names/handles + component slugs) -> existing nodes.
  for (const hit of resolver.gazetteerMentions(text)) {
    byId.set(hit.id, { id: hit.id, label: hit.label });
  }
  // @handle -> Person (ensure node exists, unknown handles included).
  for (const m of text.matchAll(HANDLE_RE)) {
    const handle = m[1];
    const id = personId(handle);
    byId.set(id, {
      id,
      label: "Person",
      ensure: { id, name: resolver.personName(handle) },
    });
  }
  // PROJ-\d+ -> Ticket (link only if the ticket exists; mergeEdge is no-op else).
  for (const m of text.matchAll(TICKET_RE)) {
    const id = `ticket:${m[0]}`;
    byId.set(id, { id, label: "Ticket" });
  }
  // URLs -> generic Entity (ensure node).
  for (const m of text.matchAll(URL_RE)) {
    const url = m[0];
    const id = entityId(url);
    byId.set(id, { id, label: "Entity", ensure: { id, name: url } });
  }

  // Don't self-mention the owner.
  byId.delete(rec.id);
  return [...byId.values()];
}

// ============================================================
// Rule-based write passes
// ============================================================

/** Phase A: MERGE the record's root node (props only). */
export async function writeRoot(tx: GraphTransaction, rec: RawRecord): Promise<void> {
  await mergeNode(tx, rec.label, rec.id, rec.props);
}

/** MERGE the known controlled-vocabulary Person/Component nodes (with names). */
export async function ensureVocab(
  tx: GraphTransaction,
  people: { handle: string; name: string }[],
  components: { slug: string; name: string }[],
): Promise<void> {
  for (const p of people) {
    const id = personId(p.handle);
    await mergeNode(tx, "Person", id, { id, name: p.name });
  }
  for (const c of components) {
    const id = `component:${slug(c.slug)}`;
    await mergeNode(tx, "Component", id, { id, name: c.name });
  }
}

/**
 * Phase B: write the record's chunks (card + body), HAS_CHUNK edges, structured
 * edges (per the schema modules), and MENTIONS edges (Chunk -> Entity/Person/...).
 * All targets are expected to exist (roots + vocab created in phase A); missing
 * targets simply produce no edge.
 */
export async function writeChunksAndStructure(
  tx: GraphTransaction,
  rec: RawRecord,
  resolver: Resolver,
): Promise<void> {
  // Chunks + HAS_CHUNK
  const chunks = buildChunks(rec);
  for (const c of chunks) {
    await mergeNode(tx, "Chunk", c.id, {
      id: c.id,
      content: c.content,
      source: c.source,
    });
    await mergeEdge(tx, rec.label, rec.id, "HAS_CHUNK", "Chunk", c.id, {
      index: c.index,
    });
  }

  // Structured edges (AUTHORED_BY/ABOUT/REFERENCES/SENT_BY/REPLY_TO/ASSIGNED/...)
  for (const ref of rec.refs) {
    await mergeEdge(
      tx,
      rec.label,
      rec.id,
      ref.rel,
      REL_TARGET_LABEL[ref.rel],
      ref.targetId,
    );
  }

  // MENTIONS hang off the primary card chunk (index 0).
  const primary = chunkId(rec.id, 0);
  for (const t of collectMentions(rec, resolver)) {
    if (t.ensure) await mergeNode(tx, t.label, t.id, t.ensure);
    await mergeEdge(tx, "Chunk", primary, "MENTIONS", t.label, t.id);
  }
}

// ============================================================
// LLM-extraction lever (optimizer-insertable, DEFAULT OFF, metered)
// ============================================================

export interface SchemaJson {
  labels: string[];
  relationships: { type: string; from: string; to: string }[];
  nodeProps?: Record<string, Record<string, string>>;
}

export interface LlmEntity {
  /** resolved canonical id (reuses Person/Component when the name is known) */
  id: string;
  label: string;
  name: string;
}
export interface LlmRelation {
  fromId: string;
  type: string;
  toId: string;
}
export interface LlmExtraction {
  entities: LlmEntity[];
  relations: LlmRelation[];
}

function sanitizeRelType(t: string): string {
  return t.toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_|_$/g, "");
}

/**
 * Call the (injected) LLM client to extract schema-constrained entities +
 * relations from a single text, metering tokens/usd. Entities are deduped
 * against the resolver's canonical Person/Component ids (so the lever reuses
 * existing nodes rather than duplicating them); unknown entities become typed
 * `Entity`s. Off-schema labels are dropped. NO network call happens unless a
 * real client is injected — the seed path never calls this.
 */
export async function extractWithLLM(
  text: string,
  schema: SchemaJson,
  llm: LlmClient,
  meter: Meter,
  field: string,
  resolver: Resolver,
): Promise<LlmExtraction> {
  const allowedLabels = new Set(schema.labels);

  const res = await llm.complete({
    field,
    system:
      "You extract a knowledge graph from text. Return STRICT JSON: " +
      `{"entities":[{"name":string,"type":string}],"relations":[{"from":string,"type":string,"to":string}]}. ` +
      "Use only entity types from this allowed set: " +
      [...allowedLabels].join(", ") +
      ". Relation 'from'/'to' must be entity names you listed.",
    user: text,
  });
  meter.record(field, res.tokens, res.usd);

  const raw = (res.json ?? {}) as {
    entities?: { name?: string; type?: string }[];
    relations?: { from?: string; type?: string; to?: string }[];
  };

  // Resolve names -> canonical ids (reuse known Person/Component).
  const idByName = new Map<string, LlmEntity>();
  const entities: LlmEntity[] = [];
  for (const e of raw.entities ?? []) {
    const name = (e.name ?? "").trim();
    if (!name) continue;
    const known = resolver.gazetteerMentions(name)[0];
    let ent: LlmEntity;
    if (known) {
      ent = { id: known.id, label: known.label, name: known.name };
    } else {
      // Unknown Person/Component names downgrade to a generic Entity: they get
      // entity:<slug> ids, so writing them under the gazetteer labels would
      // break the person:/component: id scheme — and writeLlmExtraction would
      // skip them entirely, silently losing their relations.
      const type =
        e.type && allowedLabels.has(e.type) && e.type !== "Person" && e.type !== "Component"
          ? e.type
          : "Entity";
      ent = { id: entityId(name), label: type, name };
    }
    idByName.set(name.toLowerCase(), ent);
    if (!entities.some((x) => x.id === ent.id)) entities.push(ent);
  }

  const relations: LlmRelation[] = [];
  for (const r of raw.relations ?? []) {
    const from = idByName.get((r.from ?? "").toLowerCase());
    const to = idByName.get((r.to ?? "").toLowerCase());
    const type = sanitizeRelType(r.type ?? "");
    if (!from || !to || !type) continue;
    relations.push({ fromId: from.id, type, toId: to.id });
  }

  return { entities, relations };
}

/**
 * Persist an LLM extraction via MERGE-upserts. New typed Entities get the
 * `Entity` base label plus their type label; LLM relations carry `{sources}`.
 */
export async function writeLlmExtraction(
  tx: GraphTransaction,
  ex: LlmExtraction,
  sourceField: string,
): Promise<void> {
  for (const e of ex.entities) {
    // Reused Person/Component nodes already exist; only create new Entities here.
    if (e.label === "Person" || e.label === "Component") continue;
    const labels = e.label === "Entity" ? "Entity" : `Entity:${e.label}`;
    await mergeNode(tx, labels, e.id, { id: e.id, name: e.name });
  }
  for (const r of ex.relations) {
    await tx.run(
      `MATCH (a {id: $from}), (b {id: $to})
       MERGE (a)-[rel:${sanitizeRelType(r.type)}]->(b)
       ON CREATE SET rel.sources = [$src]
       ON MATCH  SET rel.sources =
         CASE WHEN $src IN coalesce(rel.sources, []) THEN rel.sources
              ELSE coalesce(rel.sources, []) + $src END`,
      { from: r.fromId, to: r.toId, src: sourceField },
    );
  }
}
