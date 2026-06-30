// ============================================================
// ingest.ts — corpus -> Neo4j graph entrypoint + CLI
// ============================================================
//
//   npx tsx graphmod/src/ingestion/ingest.ts <corpusDir> [--db <name>] [--hash <h>]
//
// Reads graphsearch/data/corpus/{docs/*.md, chat.json, tickets.json}, builds the
// graph with idempotent MERGE-upserts (re-ingest is a no-op), creates the id
// uniqueness constraints + the `corpus_ft` fulltext index, writes
// ingest_cost.json, validates each root, and emits an IngestReport to stdout.
// Exits non-zero on ANY build/schema error so the Python reward adapter turns it
// into crashed_frac and rejects the candidate.

import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import type { GraphSession } from "../core/types";
import { CORPUS_MODULES } from "../modules/Corpus/index";
import {
  Resolver,
  KNOWN_PEOPLE,
  KNOWN_COMPONENTS,
  type RawRecord,
} from "./resolve";
import { loadMarkdownDir } from "./loaders/markdown";
import { loadChatFile } from "./loaders/chat";
import { loadJiraFile } from "./loaders/jira";
import {
  ensureVocab,
  writeRoot,
  writeChunksAndStructure,
  extractWithLLM,
  writeLlmExtraction,
  type SchemaJson,
} from "./extract";
import { Meter, getLlmClient, type LlmClient } from "./llm";

// ---- Public types ----

export interface IngestOptions {
  /** Neo4j database name override (per-candidate isolation: graph_<hash>). */
  db?: string;
  /** Ingest hash — names the cost file and the candidate DB. */
  hash?: string;
  /** Master switch for the LLM-extraction lever. DEFAULT OFF. */
  llmExtraction?: boolean;
  /** Per-record field allowlist; when set, only these fields use the lever. */
  llmFields?: string[];
  /** Injectable LLM client (tests pass a fake; default is offline-safe). */
  llm?: LlmClient;
  /** Run the post-build invariant check (every root has a card chunk). Default true. */
  validate?: boolean;
  /** Wipe all nodes/rels before building (candidate isolation on a shared DB). */
  wipe?: boolean;
}

export interface IngestReport {
  nodes: number;
  rels: number;
  byLabel: Record<string, number>;
  costPath: string;
}

const NODE_LABELS = [
  "Person",
  "Component",
  "Entity",
  "Chunk",
  "Document",
  "Message",
  "Ticket",
];

/** graphmod/ root directory (this file lives at graphmod/src/ingestion/). */
function graphmodDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..");
}

/** Neo4j Integer | number -> number (counts are small, so .low suffices). */
function toInt(v: unknown): number {
  if (typeof v === "number") return v;
  if (v && typeof v === "object" && "low" in (v as any)) return (v as any).low;
  return Number(v ?? 0);
}

// ---- Schema setup ----

/** Create id uniqueness constraints + the corpus_ft fulltext index. */
async function createSchema(session: GraphSession): Promise<void> {
  for (const label of NODE_LABELS) {
    await session.run(
      `CREATE CONSTRAINT ${label.toLowerCase()}_id IF NOT EXISTS
       FOR (n:${label}) REQUIRE n.id IS UNIQUE`,
    );
  }
  await session.run(
    `CREATE FULLTEXT INDEX corpus_ft IF NOT EXISTS
     FOR (n:Chunk) ON EACH [n.content]`,
  );
}

/** Derive the allowed-label/relationship schema for the LLM lever from the modules. */
function deriveSchemaJson(): SchemaJson {
  const here = graphmodDir();
  const file = join(here, "schema.json");
  if (existsSync(file)) {
    try {
      return JSON.parse(readFileSync(file, "utf8")) as SchemaJson;
    } catch {
      /* fall through to module-derived */
    }
  }
  const labels = new Set<string>();
  for (const mod of CORPUS_MODULES) {
    for (const def of Object.values(mod.schema.nodes)) {
      for (const l of (def as { labels: string[] }).labels) labels.add(l);
    }
  }
  return { labels: [...labels].sort(), relationships: [] };
}

// ---- Loading ----

/** Load all three sources from `corpusDir` into RawRecords. */
export function loadCorpus(corpusDir: string, resolver: Resolver): RawRecord[] {
  const records: RawRecord[] = [];
  const docsDir = join(corpusDir, "docs");
  if (existsSync(docsDir)) records.push(...loadMarkdownDir(docsDir, resolver));
  const chat = join(corpusDir, "chat.json");
  if (existsSync(chat)) records.push(...loadChatFile(chat, resolver));
  const tickets = join(corpusDir, "tickets.json");
  if (existsSync(tickets)) records.push(...loadJiraFile(tickets, resolver));
  return records;
}

/** The cost-log field id for a record (matches the CONTRACT example `chat.msg:eng:7`). */
function fieldOf(rec: RawRecord): string {
  return `${rec.kind}.${rec.id}`;
}

/**
 * Correct, cheap post-build invariant check: every Document/Message/Ticket root
 * must exist and own at least one HAS_CHUNK card chunk (the CONTRACT "card"
 * requirement that makes judge-free answering possible). Throws on any miss.
 */
async function validateBuild(
  session: GraphSession,
  records: RawRecord[],
): Promise<void> {
  const errors: string[] = [];
  for (const rec of records) {
    const res = await session.run(
      `OPTIONAL MATCH (o:${rec.label} {id: $id})
       OPTIONAL MATCH (o)-[:HAS_CHUNK]->(c:Chunk)
       RETURN count(DISTINCT o) AS hasRoot, count(DISTINCT c) AS chunks`,
      { id: rec.id },
    );
    const row = res.records[0];
    const hasRoot = toInt(row.get("hasRoot"));
    const chunks = toInt(row.get("chunks"));
    if (hasRoot < 1) errors.push(`${rec.id}: root node missing after write`);
    else if (chunks < 1)
      errors.push(`${rec.id}: no HAS_CHUNK card chunk (violates card requirement)`);
  }
  if (errors.length) {
    throw new Error("post-ingest validation failed:\n  - " + errors.join("\n  - "));
  }
}

// ---- Main ----

/**
 * Build the corpus graph in `session` (idempotent). Returns an IngestReport.
 * Throws on any schema/validation error — the CLI maps that to exit code 1.
 */
export async function ingest(
  session: GraphSession,
  corpusDir: string,
  opts: IngestOptions = {},
): Promise<IngestReport> {
  const resolver = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);
  const meter = new Meter(opts.hash ?? "seed");

  // 0) Optional wipe — candidate isolation on a shared (Neo4j Community single) DB.
  //    Each candidate's ingestion must build a fresh graph; MERGE alone would layer
  //    a new candidate's nodes on top of the previous one. Constraints/indexes are
  //    not nodes, so they survive (and createSchema is IF NOT EXISTS regardless).
  if (opts.wipe) {
    await session.run(`MATCH (n) DETACH DELETE n`);
  }

  // 1) Constraints + fulltext index.
  await createSchema(session);

  // 2) Load every source into the shared RawRecord IR.
  const records = loadCorpus(corpusDir, resolver);

  // 3) Rule-based build — one transaction for the whole tiny corpus.
  //    Phase A: vocab + all root nodes (so every edge target exists with full
  //    props before any edge/forward-reference is written). Phase B: chunks,
  //    HAS_CHUNK, structured edges, MENTIONS.
  await session.executeWrite(async (tx) => {
    await ensureVocab(tx, KNOWN_PEOPLE, KNOWN_COMPONENTS);
    for (const rec of records) await writeRoot(tx, rec);
    for (const rec of records) await writeChunksAndStructure(tx, rec, resolver);
  });

  // 4) LLM-extraction lever — DEFAULT OFF. Gated by opts.llmExtraction (and an
  //    optional per-field allowlist). When off, the meter stays {0,0,[]} and no
  //    LLM client is ever touched, so the seed run never makes a network call.
  if (opts.llmExtraction) {
    const llm = opts.llm ?? getLlmClient();
    const schema = deriveSchemaJson();
    for (const rec of records) {
      const field = fieldOf(rec);
      if (opts.llmFields && !opts.llmFields.includes(field)) continue;
      const ex = await extractWithLLM(rec.text, schema, llm, meter, field, resolver);
      await session.executeWrite((tx) => writeLlmExtraction(tx, ex, field));
    }
  }

  // 5) Validate the build. NOTE: we deliberately do NOT call the per-root
  //    Module.assertValid here. buildTraversalQuery expands each relationship a
  //    single hop from its anchor, so the *target* of a self-referential edge
  //    (REFERENCES Document->Document, BLOCKS Ticket->Ticket, REPLY_TO
  //    Message->Message — all present in this corpus) is tagged with the root's
  //    node kind and then required to satisfy HAS_CHUNK 1..n, even though its
  //    chunks were never fetched by that one-hop traversal. That yields a FALSE
  //    SchemaValidationError on a perfectly valid graph. Instead we assert the
  //    invariant that actually matters (and is correctly checkable): every root
  //    exists with at least one HAS_CHUNK "card" chunk. Any violation throws and
  //    the CLI exits non-zero — a bad write is never allowed to pass silently.
  if (opts.validate !== false) {
    await validateBuild(session, records);
  }

  // 6) Cost file.
  const costPath = meter.writeCostFiles(graphmodDir());

  // 7) Counts for the report.
  const nodes = toInt((await session.run(`MATCH (n) RETURN count(n) AS c`)).records[0].get("c"));
  const rels = toInt((await session.run(`MATCH ()-[r]->() RETURN count(r) AS c`)).records[0].get("c"));
  const byLabel: Record<string, number> = {};
  const labRes = await session.run(
    `MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS c`,
  );
  for (const r of labRes.records) byLabel[r.get("label")] = toInt(r.get("c"));

  return { nodes, rels, byLabel, costPath };
}

// ---- CLI ----

function parseArgs(argv: string[]): { corpusDir: string; opts: IngestOptions } {
  const positional: string[] = [];
  const opts: IngestOptions = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--db") opts.db = argv[++i];
    else if (a === "--hash") opts.hash = argv[++i];
    else if (a === "--llm") opts.llmExtraction = true;
    else if (a === "--wipe") opts.wipe = true;
    else if (a === "--no-validate") opts.validate = false;
    else positional.push(a);
  }
  const corpusDir = positional[0];
  if (!corpusDir) throw new Error("usage: ingest.ts <corpusDir> [--db <name>] [--hash <h>]");
  return { corpusDir, opts };
}

async function main(): Promise<void> {
  const { corpusDir, opts } = parseArgs(process.argv.slice(2));
  // Honor --db by setting NEO4J_DB before getSession() reads it.
  if (opts.db) process.env.NEO4J_DB = opts.db;

  // Imported lazily so module load (and tests) don't require a live driver.
  const { getSession } = await import("../../examples/session");
  const session = await getSession();
  let report: IngestReport;
  try {
    report = await ingest(session, corpusDir, opts);
  } finally {
    await (session as unknown as { close?: () => Promise<void> }).close?.();
  }
  // Flush the report fully before the caller (the Python reward adapter) reads it.
  await new Promise<void>((resolve) =>
    process.stdout.write(JSON.stringify(report) + "\n", () => resolve()),
  );
}

// Run as a script (tsx). Any error -> non-zero exit with the message surfaced.
const isMain = (() => {
  try {
    return fileURLToPath(import.meta.url) === process.argv[1];
  } catch {
    return false;
  }
})();

if (isMain) {
  // getSession() (examples/session.ts) creates a neo4j-driver Driver but returns
  // only the Session, so the driver's connection pool is never closed and keeps
  // Node's event loop alive — the process would hang forever after printing the
  // report. Force a clean exit once main() has flushed stdout + closed the session.
  main()
    .then(() => process.exit(0))
    .catch((err) => {
      process.stderr.write(
        `ingest failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`,
      );
      process.exit(1);
    });
}
