// ============================================================
// videoFile.delete.test.ts — VideoFileModule.delete
// ============================================================
// Run me:  npx tsx --test -r dotenv/config tests/Module/Document/VideoFile/videoFile.delete.test.ts
//          (or the ▶ gutter icon next to each `it(...)`)
//
// Graph shape under test (VideoFileModule schema):
//
//   (:Person) ──SPEAKS_IN {id, role}──▶ (:Chunk)
//                                          ▲
//   (:Document:VideoFile) ──HAS_CHUNK ─────┘──MENTIONS──▶ (:Entity)
//
// Ownership semantics encoded in onDelete:
//   owned  (Document, Chunk) — deleted with the module
//   shared (Entity)          — deleted ONLY when fully orphaned
//   shared (Person)          — NEVER deleted; only its SPEAKS_IN edge is cut
//                              (DETACH DELETE of the chunk removes the edge)
//
// All ids are namespaced under the `vf:` prefix so the suite scopes every
// query/cleanup to its own data.
//
import { describe, it, before, beforeEach, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../../_db";
import type { GraphSession } from "../../../../src/index";
import { VideoFileModule } from "../../../../src/modules/Document/VideoFile/videoFile.module";

const PREFIX = "vf:";

//   pAlice ─SPEAKS_IN─▶ cA0 ─MENTIONS─▶ eShared      pBob ─SPEAKS_IN─▶ cB0 ─▶ eShared
//   pAlice ─SPEAKS_IN─▶ cA1 ─MENTIONS─▶ eOnlyA                              └▶ eOnlyB
//   docA ─HAS_CHUNK─▶ cA0, cA1                       docB ─HAS_CHUNK─▶ cB0
//
// Deleting docA must remove docA + cA0 + cA1 + eOnlyA, keep eShared (still
// mentioned by docB), and keep BOTH persons (shared) while cutting pAlice's
// SPEAKS_IN edges to the now-deleted chunks.
const ID = {
  docA: "vf:doc-a",
  cA0: "vf:doc-a-chunk-0",
  cA1: "vf:doc-a-chunk-1",
  docB: "vf:doc-b",
  cB0: "vf:doc-b-chunk-0",
  eShared: "vf:entity-shared",
  eOnlyA: "vf:entity-only-a",
  eOnlyB: "vf:entity-only-b",
  pAlice: "vf:person-alice",
  pBob: "vf:person-bob",
} as const;

async function clear(session: GraphSession): Promise<void> {
  await session.run(`MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n`, {
    prefix: PREFIX,
  });
}

async function seed(session: GraphSession): Promise<void> {
  await session.run(
    `
    CREATE (docA:Document:VideoFile {id: $docA})
    CREATE (cA0:Chunk {id: $cA0, content: 'A0 — Alice on the shared entity'})
    CREATE (cA1:Chunk {id: $cA1, content: 'A1 — Alice on an A-only entity'})
    CREATE (docB:Document:VideoFile {id: $docB})
    CREATE (cB0:Chunk {id: $cB0, content: 'B0 — Bob on shared + B-only entities'})
    CREATE (eShared:Entity {id: $eShared})
    CREATE (eOnlyA:Entity {id: $eOnlyA})
    CREATE (eOnlyB:Entity {id: $eOnlyB})
    CREATE (pAlice:Person {id: $pAlice, name: 'Alice Müller'})
    CREATE (pBob:Person {id: $pBob, name: 'Bob Schmidt'})

    CREATE (docA)-[:HAS_CHUNK {index: 0}]->(cA0)
    CREATE (docA)-[:HAS_CHUNK {index: 1}]->(cA1)
    CREATE (docB)-[:HAS_CHUNK {index: 0}]->(cB0)

    CREATE (pAlice)-[:SPEAKS_IN {id: 'vf:spk-a0', role: 'host'}]->(cA0)
    CREATE (pAlice)-[:SPEAKS_IN {id: 'vf:spk-a1', role: 'host'}]->(cA1)
    CREATE (pBob)-[:SPEAKS_IN {id: 'vf:spk-b0', role: 'guest'}]->(cB0)

    CREATE (cA0)-[:MENTIONS]->(eShared)
    CREATE (cA1)-[:MENTIONS]->(eOnlyA)
    CREATE (cB0)-[:MENTIONS]->(eShared)
    CREATE (cB0)-[:MENTIONS]->(eOnlyB)
    `,
    ID,
  );
}

async function exists(session: GraphSession, id: string): Promise<boolean> {
  const res = await session.run(`MATCH (n {id: $id}) RETURN count(n) AS c`, { id });
  return res.records[0].get("c").toNumber() > 0;
}

describe("VideoFileModule — seeded data", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  beforeEach(() => clear(session));
  after(async () => {
    await clear(session);
    await closeDriver();
  });

  it("hydrates the document, its chunks, and reverse SPEAKS_IN speakers", async () => {
    await seed(session);

    const doc = await VideoFileModule.fromLazyGraph(session, ID.docA);
    assert.equal(doc.id, ID.docA);

    const chunks = await doc.HAS_CHUNK;
    assert.equal(chunks.length, 2, "docA should have 2 chunks");

    // SPEAKS_IN is an inbound edge: reach the speaker via `_reverse`.
    const speakers = await doc.HAS_CHUNK.at(0)._reverse.SPEAKS_IN;
    assert.equal(speakers.length, 1);
    assert.equal((speakers[0] as any).id, ID.pAlice);

    const role = await doc.HAS_CHUNK.at(0)._reverse.SPEAKS_IN.at(0)._rel.role;
    assert.equal(role, "host");
  });

  it("seeds Alice speaking in both of docA's chunks", async () => {
    await seed(session);
    const res = await session.run(
      `MATCH (p:Person {id: $id})-[:SPEAKS_IN]->(:Chunk) RETURN count(*) AS c`,
      { id: ID.pAlice },
    );
    assert.equal(res.records[0].get("c").toNumber(), 2);
  });
});

describe("VideoFileModule.delete", () => {
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

  it("deletes the root document node", async () => {
    await VideoFileModule.delete(session, ID.docA);
    assert.equal(await exists(session, ID.docA), false);
  });

  it("deletes every chunk owned by the document — no orphaned chunks", async () => {
    await VideoFileModule.delete(session, ID.docA);

    assert.equal(await exists(session, ID.cA0), false);
    assert.equal(await exists(session, ID.cA1), false);

    const orphans = await session.run(
      `MATCH (c:Chunk) WHERE c.id STARTS WITH $prefix AND NOT ()-[:HAS_CHUNK]->(c)
       RETURN collect(c.id) AS ids`,
      { prefix: PREFIX },
    );
    assert.deepEqual(orphans.records[0].get("ids"), [], "expected no orphaned chunks");
  });

  it("deletes entities that become fully orphaned by the delete", async () => {
    await VideoFileModule.delete(session, ID.docA);
    assert.equal(await exists(session, ID.eOnlyA), false);
  });

  it("preserves a shared entity still mentioned by another document", async () => {
    await VideoFileModule.delete(session, ID.docA);
    assert.equal(await exists(session, ID.eShared), true);

    const res = await session.run(
      `MATCH (c:Chunk)-[:MENTIONS]->(:Entity {id: $id}) RETURN collect(c.id) AS ids`,
      { id: ID.eShared },
    );
    assert.deepEqual(res.records[0].get("ids"), [ID.cB0]);
  });

  it("NEVER deletes shared Person nodes — only their SPEAKS_IN edges are cut", async () => {
    await VideoFileModule.delete(session, ID.docA);

    // Both speakers survive: Alice spoke only in docA, Bob only in docB.
    assert.equal(await exists(session, ID.pAlice), true, "Alice must survive the delete");
    assert.equal(await exists(session, ID.pBob), true, "Bob untouched");

    // Alice's SPEAKS_IN edges to the deleted chunks are gone (she's now edgeless).
    const res = await session.run(
      `MATCH (p:Person {id: $id})-[r:SPEAKS_IN]->() RETURN count(r) AS c`,
      { id: ID.pAlice },
    );
    assert.equal(res.records[0].get("c").toNumber(), 0, "Alice should have no dangling SPEAKS_IN edges");
  });

  it("leaves the other document's subgraph (incl. its speaker) intact", async () => {
    await VideoFileModule.delete(session, ID.docA);

    assert.equal(await exists(session, ID.docB), true);
    assert.equal(await exists(session, ID.cB0), true);
    assert.equal(await exists(session, ID.eOnlyB), true);

    const doc = await VideoFileModule.fromLazyGraph(session, ID.docB);
    const speakers = await doc.HAS_CHUNK.at(0)._reverse.SPEAKS_IN;
    assert.equal(speakers.length, 1);
    assert.equal((speakers[0] as any).id, ID.pBob, "Bob still speaks in docB");
  });

  it("throws when deleting a document that no longer exists — and changes nothing", async () => {
    await VideoFileModule.delete(session, ID.docA);

    // `delete` hydrates the root first, so a second delete rejects (the root
    // node is gone). It is NOT a silent no-op — callers must guard existence.
    await assert.rejects(
      () => VideoFileModule.delete(session, ID.docA),
      /No node "Document" with id "vf:doc-a"/,
    );

    assert.equal(await exists(session, ID.eShared), true);
    assert.equal(await exists(session, ID.pAlice), true);
    assert.equal(await exists(session, ID.docB), true);
  });
});
