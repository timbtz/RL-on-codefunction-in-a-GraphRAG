// ============================================================
// videoFile.large.test.ts — VideoFileModule.delete at scale
// ============================================================
// Run me:  npx tsx --test -r dotenv/config tests/Module/Document/VideoFile/videoFile.large.test.ts
//          (or the ▶ gutter icon next to each `it(...)`)
//
// Same module + ownership semantics as videoFile.delete.test.ts, but stressed
// with many documents, chunks, entities AND speakers so the "Person is shared
// and NEVER deleted, only its SPEAKS_IN edge is cut" rule is exercised across a
// large, interconnected graph.
//
// Generated shape (all ids under the `vfL:` prefix):
//
//   (:Person) ──SPEAKS_IN {id, role}──▶ (:Chunk)
//                                          ▲
//   (:Document:VideoFile) ──HAS_CHUNK ─────┘──MENTIONS──▶ (:Entity)
//
//   • per-chunk-only entities — orphaned (and deleted) with their document.
//   • global entities — entity k mentioned by chunk k of every document → shared.
//   • speakers — a small pool of PERSON_COUNT people, each speaking in chunks of
//     MANY documents. Deleting one document cuts only their edges to that doc's
//     chunks; the Person nodes themselves always survive.
//
import { describe, it, before, beforeEach, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../../_db";
import type { GraphSession } from "../../../../src/index";
import { VideoFileModule } from "../../../../src/modules/Document/VideoFile/videoFile.module";

const PREFIX = "vfL:";

const DOC_COUNT = 6;
const CHUNKS_PER_DOC = 20;
const GLOBAL_ENTITIES = 4; // must be <= CHUNKS_PER_DOC
const PERSON_COUNT = 5;

const TOTAL_CHUNKS = DOC_COUNT * CHUNKS_PER_DOC; // 120
const TOTAL_PRIVATE_ENTITIES = DOC_COUNT * CHUNKS_PER_DOC; // 120

const docId = (i: number) => `${PREFIX}doc-${i}`;
const chunkId = (i: number, j: number) => `${PREFIX}doc-${i}-chunk-${j}`;
const privateEntId = (i: number, j: number) => `${PREFIX}ent-d${i}-c${j}`;
const globalEntId = (k: number) => `${PREFIX}ent-global-${k}`;
const personId = (p: number) => `${PREFIX}person-${p}`;
const speaksId = (i: number, j: number) => `${PREFIX}spk-d${i}-c${j}`;

// Every chunk has exactly one speaker, assigned round-robin across the pool,
// so each person ends up speaking across multiple documents.
const speakerForChunk = (i: number, j: number) => (i * CHUNKS_PER_DOC + j) % PERSON_COUNT;

async function clear(session: GraphSession): Promise<void> {
  await session.run(`MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n`, {
    prefix: PREFIX,
  });
}

async function seed(session: GraphSession): Promise<void> {
  const docs: Array<{ id: string }> = [];
  const persons: Array<{ id: string; name: string }> = [];
  const chunks: Array<{ id: string; docId: string; index: number; content: string }> = [];
  const entities: Array<{ id: string }> = [];
  const mentions: Array<{ chunkId: string; entId: string }> = [];
  const speaks: Array<{ personId: string; chunkId: string; id: string; role: string }> = [];

  for (let p = 0; p < PERSON_COUNT; p++) persons.push({ id: personId(p), name: `Speaker ${p}` });
  for (let k = 0; k < GLOBAL_ENTITIES; k++) entities.push({ id: globalEntId(k) });

  for (let i = 0; i < DOC_COUNT; i++) {
    docs.push({ id: docId(i) });
    for (let j = 0; j < CHUNKS_PER_DOC; j++) {
      const cid = chunkId(i, j);
      chunks.push({ id: cid, docId: docId(i), index: j, content: `doc ${i} / chunk ${j}` });

      const pe = privateEntId(i, j);
      entities.push({ id: pe });
      mentions.push({ chunkId: cid, entId: pe });
      if (j < GLOBAL_ENTITIES) mentions.push({ chunkId: cid, entId: globalEntId(j) });

      const p = speakerForChunk(i, j);
      speaks.push({
        personId: personId(p),
        chunkId: cid,
        id: speaksId(i, j),
        role: j === 0 ? "host" : "guest",
      });
    }
  }

  await session.run(`UNWIND $docs AS d CREATE (:Document:VideoFile {id: d.id})`, { docs });
  await session.run(`UNWIND $persons AS p CREATE (:Person {id: p.id, name: p.name})`, { persons });
  await session.run(`UNWIND $entities AS e CREATE (:Entity {id: e.id})`, { entities });
  await session.run(
    `UNWIND $chunks AS ch
     MATCH (doc:Document:VideoFile {id: ch.docId})
     CREATE (doc)-[:HAS_CHUNK {index: ch.index}]->(:Chunk {id: ch.id, content: ch.content})`,
    { chunks },
  );
  await session.run(
    `UNWIND $mentions AS m
     MATCH (c:Chunk {id: m.chunkId}), (e:Entity {id: m.entId})
     CREATE (c)-[:MENTIONS]->(e)`,
    { mentions },
  );
  await session.run(
    `UNWIND $speaks AS s
     MATCH (p:Person {id: s.personId}), (c:Chunk {id: s.chunkId})
     CREATE (p)-[:SPEAKS_IN {id: s.id, role: s.role}]->(c)`,
    { speaks },
  );
}

async function count(session: GraphSession, cypher: string): Promise<number> {
  const res = await session.run(cypher, { prefix: PREFIX });
  return res.records[0].get("c").toNumber();
}

const nodeCount = (s: GraphSession, label: string) =>
  count(s, `MATCH (n:${label}) WHERE n.id STARTS WITH $prefix RETURN count(n) AS c`);

async function exists(session: GraphSession, id: string): Promise<boolean> {
  const res = await session.run(`MATCH (n {id: $id}) RETURN count(n) AS c`, { id });
  return res.records[0].get("c").toNumber() > 0;
}

describe("VideoFileModule — large dataset seed", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  beforeEach(async () => {
    await clear(session);
    await seed(session);
  });
  after(async () => {
    await clear(session);
    await closeDriver();
  });

  it("inserts every document, chunk, entity and person", async () => {
    assert.equal(await nodeCount(session, "Document"), DOC_COUNT);
    assert.equal(await nodeCount(session, "Chunk"), TOTAL_CHUNKS);
    assert.equal(await nodeCount(session, "Entity"), TOTAL_PRIVATE_ENTITIES + GLOBAL_ENTITIES);
    assert.equal(await nodeCount(session, "Person"), PERSON_COUNT);
  });

  it("wires one SPEAKS_IN edge per chunk", async () => {
    const speaks = await count(
      session,
      `MATCH (:Person)-[r:SPEAKS_IN]->(c:Chunk) WHERE c.id STARTS WITH $prefix RETURN count(r) AS c`,
    );
    assert.equal(speaks, TOTAL_CHUNKS);
  });

  it("spreads each speaker across multiple documents", async () => {
    // Speakers are assigned round-robin, so each appears in chunks of many docs.
    const docsForP0 = await session.run(
      `MATCH (p:Person {id: $id})-[:SPEAKS_IN]->(:Chunk)<-[:HAS_CHUNK]-(d:Document)
       RETURN count(DISTINCT d) AS c`,
      { id: personId(0) },
    );
    assert.ok(
      docsForP0.records[0].get("c").toNumber() > 1,
      "speaker 0 should appear in more than one document",
    );
  });
});

describe("VideoFileModule.delete — large dataset", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  beforeEach(async () => {
    await clear(session);
    await seed(session);
  });
  after(async () => {
    await clear(session);
    await closeDriver();
  });

  it("deletes one document, its chunks, and its private entities", async () => {
    await VideoFileModule.delete(session, docId(0));

    assert.equal(await exists(session, docId(0)), false);
    assert.equal(await nodeCount(session, "Document"), DOC_COUNT - 1);
    assert.equal(await nodeCount(session, "Chunk"), TOTAL_CHUNKS - CHUNKS_PER_DOC);

    for (let j = 0; j < CHUNKS_PER_DOC; j++) {
      assert.equal(await exists(session, privateEntId(0, j)), false);
    }
  });

  it("NEVER deletes any Person, even when their chunks are gone", async () => {
    await VideoFileModule.delete(session, docId(0));

    // all speakers survive
    assert.equal(await nodeCount(session, "Person"), PERSON_COUNT);
    for (let p = 0; p < PERSON_COUNT; p++) {
      assert.equal(await exists(session, personId(p)), true, `person ${p} must survive`);
    }

    // doc 0's SPEAKS_IN edges are cut: no person still speaks in a (now deleted) doc-0 chunk
    const danglingToDoc0 = await count(
      session,
      `MATCH (:Person)-[r:SPEAKS_IN]->(c:Chunk) WHERE c.id STARTS WITH '${PREFIX}doc-0-'
       RETURN count(r) AS c`,
    );
    assert.equal(danglingToDoc0, 0, "no SPEAKS_IN edge should point at a deleted chunk");

    // total SPEAKS_IN edges drop by exactly doc 0's chunk count
    const remaining = await count(
      session,
      `MATCH (:Person)-[r:SPEAKS_IN]->(c:Chunk) WHERE c.id STARTS WITH $prefix RETURN count(r) AS c`,
    );
    assert.equal(remaining, TOTAL_CHUNKS - CHUNKS_PER_DOC);
  });

  it("leaves no orphaned chunks or orphaned entities", async () => {
    await VideoFileModule.delete(session, docId(0));

    const orphanChunks = await count(
      session,
      `MATCH (c:Chunk) WHERE c.id STARTS WITH $prefix AND NOT ()-[:HAS_CHUNK]->(c)
       RETURN count(c) AS c`,
    );
    assert.equal(orphanChunks, 0);

    const orphanEntities = await count(
      session,
      `MATCH (e:Entity) WHERE e.id STARTS WITH $prefix AND NOT (e)--()
       RETURN count(e) AS c`,
    );
    assert.equal(orphanEntities, 0);
  });

  it("preserves global entities still mentioned by other documents", async () => {
    await VideoFileModule.delete(session, docId(0));
    for (let k = 0; k < GLOBAL_ENTITIES; k++) {
      assert.equal(await exists(session, globalEntId(k)), true);
    }
  });

  it("deletes all documents but keeps every Person (edgeless), then no entities remain", async () => {
    for (let i = 0; i < DOC_COUNT; i++) {
      await VideoFileModule.delete(session, docId(i));
    }

    assert.equal(await nodeCount(session, "Document"), 0);
    assert.equal(await nodeCount(session, "Chunk"), 0);
    assert.equal(await nodeCount(session, "Entity"), 0);

    // persons remain, now with zero SPEAKS_IN edges
    assert.equal(await nodeCount(session, "Person"), PERSON_COUNT);
    const anySpeaks = await count(
      session,
      `MATCH (:Person)-[r:SPEAKS_IN]->() WHERE startNode(r).id STARTS WITH $prefix RETURN count(r) AS c`,
    );
    assert.equal(anySpeaks, 0, "all SPEAKS_IN edges should be gone with their chunks");
  });
});
