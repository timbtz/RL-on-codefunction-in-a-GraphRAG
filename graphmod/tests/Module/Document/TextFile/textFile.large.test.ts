// ============================================================
// textFile.large.test.ts — TextFileModule.delete at scale
// ============================================================
// Run me:  npx tsx --test -r dotenv/config tests/Module/Document/TextFile/textFile.large.test.ts
//          (or the ▶ gutter icon next to each `it(...)`)
//
// Same module + ownership semantics as textFile.delete.test.ts, but stressed
// with a much bigger graph so orphan-cleanup and shared-node preservation are
// exercised across many documents, chunks and entities at once.
//
// Generated shape (all ids under the `tfL:` prefix):
//
//   DOC_COUNT documents, each with CHUNKS_PER_DOC chunks:
//     (:Document:TextFile) ──HAS_CHUNK {index}──▶ (:Chunk) ──MENTIONS──▶ (:Entity)
//
//   Two flavours of entity:
//     • per-chunk-only entities — one private entity per chunk. They are
//       mentioned by exactly one chunk, so deleting that chunk's document
//       fully orphans them → they must be deleted.
//     • global entities — entity k is mentioned by chunk k of EVERY document.
//       Deleting one document leaves them mentioned by the others → they must
//       survive, until the last document mentioning them is gone.
//
import { describe, it, before, beforeEach, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../../_db";
import type { GraphSession } from "../../../../src/index";
import { TextFileModule } from "../../../../src/modules/Document/TextFile/textFile.module";

const PREFIX = "tfL:";

const DOC_COUNT = 8;
const CHUNKS_PER_DOC = 25;
const GLOBAL_ENTITIES = 5; // must be <= CHUNKS_PER_DOC

const TOTAL_CHUNKS = DOC_COUNT * CHUNKS_PER_DOC; // 200
const TOTAL_PRIVATE_ENTITIES = DOC_COUNT * CHUNKS_PER_DOC; // 200, one per chunk

const docId = (i: number) => `${PREFIX}doc-${i}`;
const chunkId = (i: number, j: number) => `${PREFIX}doc-${i}-chunk-${j}`;
const privateEntId = (i: number, j: number) => `${PREFIX}ent-d${i}-c${j}`;
const globalEntId = (k: number) => `${PREFIX}ent-global-${k}`;

async function clear(session: GraphSession): Promise<void> {
  await session.run(`MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n`, {
    prefix: PREFIX,
  });
}

/**
 * Build the dataset in JS, then bulk-insert it with a handful of UNWIND
 * passes (nodes first, then relationships) — far cheaper than one CREATE per
 * row and keeps the seed deterministic.
 */
async function seed(session: GraphSession): Promise<void> {
  const docs: Array<{ id: string }> = [];
  const chunks: Array<{ id: string; docId: string; index: number; content: string }> = [];
  const entities: Array<{ id: string }> = [];
  const mentions: Array<{ chunkId: string; entId: string }> = [];

  for (let k = 0; k < GLOBAL_ENTITIES; k++) entities.push({ id: globalEntId(k) });

  for (let i = 0; i < DOC_COUNT; i++) {
    docs.push({ id: docId(i) });
    for (let j = 0; j < CHUNKS_PER_DOC; j++) {
      const cid = chunkId(i, j);
      chunks.push({ id: cid, docId: docId(i), index: j, content: `doc ${i} / chunk ${j}` });

      // one private entity per chunk → orphaned when its document is deleted
      const pe = privateEntId(i, j);
      entities.push({ id: pe });
      mentions.push({ chunkId: cid, entId: pe });

      // chunk j (for j < GLOBAL_ENTITIES) also mentions global entity j,
      // making that entity shared across every document.
      if (j < GLOBAL_ENTITIES) mentions.push({ chunkId: cid, entId: globalEntId(j) });
    }
  }

  await session.run(`UNWIND $docs AS d CREATE (:Document:TextFile {id: d.id})`, { docs });
  await session.run(`UNWIND $entities AS e CREATE (:Entity {id: e.id})`, { entities });
  await session.run(
    `UNWIND $chunks AS ch
     MATCH (doc:Document:TextFile {id: ch.docId})
     CREATE (doc)-[:HAS_CHUNK {index: ch.index}]->(:Chunk {id: ch.id, content: ch.content})`,
    { chunks },
  );
  await session.run(
    `UNWIND $mentions AS m
     MATCH (c:Chunk {id: m.chunkId}), (e:Entity {id: m.entId})
     CREATE (c)-[:MENTIONS]->(e)`,
    { mentions },
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

describe("TextFileModule — large dataset seed", () => {
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

  it("inserts every document, chunk and entity", async () => {
    assert.equal(await nodeCount(session, "Document"), DOC_COUNT);
    assert.equal(await nodeCount(session, "Chunk"), TOTAL_CHUNKS);
    assert.equal(
      await nodeCount(session, "Entity"),
      TOTAL_PRIVATE_ENTITIES + GLOBAL_ENTITIES,
    );
  });

  it("wires the expected number of HAS_CHUNK and MENTIONS edges", async () => {
    const hasChunk = await count(
      session,
      `MATCH (d:Document)-[r:HAS_CHUNK]->(:Chunk) WHERE d.id STARTS WITH $prefix RETURN count(r) AS c`,
    );
    assert.equal(hasChunk, TOTAL_CHUNKS);

    // one MENTIONS per chunk (private) + GLOBAL_ENTITIES per document (global)
    const mentions = await count(
      session,
      `MATCH (c:Chunk)-[r:MENTIONS]->(:Entity) WHERE c.id STARTS WITH $prefix RETURN count(r) AS c`,
    );
    assert.equal(mentions, TOTAL_CHUNKS + DOC_COUNT * GLOBAL_ENTITIES);
  });

  it("makes each global entity shared across all documents", async () => {
    const res = await session.run(
      `MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity {id: $id})
       RETURN count(DISTINCT d) AS c`,
      { id: globalEntId(0) },
    );
    assert.equal(res.records[0].get("c").toNumber(), DOC_COUNT);
  });

  it("hydrates an arbitrary document with all its chunks", async () => {
    const doc = await TextFileModule.fromLazyGraph(session, docId(3));
    const chunks = await doc.HAS_CHUNK;
    assert.equal(chunks.length, CHUNKS_PER_DOC);
  });
});

describe("TextFileModule.delete — large dataset", () => {
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

  it("deletes one document, all its chunks, and all its private entities", async () => {
    await TextFileModule.delete(session, docId(0));

    assert.equal(await exists(session, docId(0)), false);
    assert.equal(await nodeCount(session, "Document"), DOC_COUNT - 1);
    assert.equal(await nodeCount(session, "Chunk"), TOTAL_CHUNKS - CHUNKS_PER_DOC);

    // every one of doc 0's private entities is now orphaned → gone
    for (let j = 0; j < CHUNKS_PER_DOC; j++) {
      assert.equal(await exists(session, privateEntId(0, j)), false);
    }
  });

  it("leaves no orphaned chunks or orphaned entities in the namespace", async () => {
    await TextFileModule.delete(session, docId(0));

    const orphanChunks = await count(
      session,
      `MATCH (c:Chunk) WHERE c.id STARTS WITH $prefix AND NOT ()-[:HAS_CHUNK]->(c)
       RETURN count(c) AS c`,
    );
    assert.equal(orphanChunks, 0, "no chunk should survive without an owning document");

    const orphanEntities = await count(
      session,
      `MATCH (e:Entity) WHERE e.id STARTS WITH $prefix AND NOT (e)--()
       RETURN count(e) AS c`,
    );
    assert.equal(orphanEntities, 0, "no entity should survive with zero relationships");
  });

  it("preserves the global entities (still mentioned by the other documents)", async () => {
    await TextFileModule.delete(session, docId(0));

    for (let k = 0; k < GLOBAL_ENTITIES; k++) {
      assert.equal(await exists(session, globalEntId(k)), true);
    }

    // global entity 0 dropped from DOC_COUNT mentioning docs to DOC_COUNT - 1
    const res = await session.run(
      `MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity {id: $id})
       RETURN count(DISTINCT d) AS c`,
      { id: globalEntId(0) },
    );
    assert.equal(res.records[0].get("c").toNumber(), DOC_COUNT - 1);
  });

  it("leaves every other document fully intact", async () => {
    await TextFileModule.delete(session, docId(0));

    for (let i = 1; i < DOC_COUNT; i++) {
      const doc = await TextFileModule.fromLazyGraph(session, docId(i));
      const chunks = await doc.HAS_CHUNK;
      assert.equal(chunks.length, CHUNKS_PER_DOC, `doc ${i} should keep all its chunks`);
    }
  });

  it("empties the whole namespace when every document is deleted in turn", async () => {
    for (let i = 0; i < DOC_COUNT; i++) {
      await TextFileModule.delete(session, docId(i));
    }

    assert.equal(await nodeCount(session, "Document"), 0);
    assert.equal(await nodeCount(session, "Chunk"), 0);
    // the global entities survive every intermediate delete and only vanish
    // once the final document mentioning them is removed.
    assert.equal(await nodeCount(session, "Entity"), 0);

    const anyLeft = await count(
      session,
      `MATCH (n) WHERE n.id STARTS WITH $prefix RETURN count(n) AS c`,
    );
    assert.equal(anyLeft, 0, "the entire tfL: namespace should be empty");
  });
});
