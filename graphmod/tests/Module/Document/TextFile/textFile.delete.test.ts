// ============================================================
// textFile.delete.test.ts — TextFileModule.delete
// ============================================================
// Run me:  npx tsx --test -r dotenv/config tests/Module/Document/TextFile/textFile.delete.test.ts
//          (or the ▶ gutter icon next to each `it(...)`)
//
// Graph shape under test (TextFileModule schema):
//
//   (:Document:TextFile) ──HAS_CHUNK {index}──▶ (:Chunk) ──MENTIONS──▶ (:Entity)
//
// Ownership semantics encoded in onDelete:
//   owned  (Document, Chunk) — deleted with the module
//   shared (Entity)          — deleted ONLY when it becomes fully orphaned
//                              (no remaining relationships after the chunks go)
//
// All ids are namespaced under the `tf:` prefix so the suite can scope every
// query/cleanup to its own data and never touch other graph contents.
//
import { describe, it, before, beforeEach, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../../_db";
import type { GraphSession } from "../../../../src/index";
import { TextFileModule } from "../../../../src/modules/Document/TextFile/textFile.module";

const PREFIX = "tf:";

// Two documents that deliberately share an entity:
//
//   docA ─▶ cA0 ─▶ eShared        docB ─▶ cB0 ─▶ eShared
//        └▶ cA1 ─▶ eOnlyA                      └▶ eOnlyB
//
// Deleting docA must remove docA + cA0 + cA1 + eOnlyA, but keep eShared
// (still mentioned by docB) and the whole docB subgraph untouched.
const ID = {
  docA: "tf:doc-a",
  cA0: "tf:doc-a-chunk-0",
  cA1: "tf:doc-a-chunk-1",
  docB: "tf:doc-b",
  cB0: "tf:doc-b-chunk-0",
  eShared: "tf:entity-shared",
  eOnlyA: "tf:entity-only-a",
  eOnlyB: "tf:entity-only-b",
} as const;

async function clear(session: GraphSession): Promise<void> {
  await session.run(`MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n`, {
    prefix: PREFIX,
  });
}

async function seed(session: GraphSession): Promise<void> {
  await session.run(
    `
    CREATE (docA:Document:TextFile {id: $docA})
    CREATE (cA0:Chunk {id: $cA0, content: 'A0 — mentions the shared entity'})
    CREATE (cA1:Chunk {id: $cA1, content: 'A1 — mentions an A-only entity'})
    CREATE (docB:Document:TextFile {id: $docB})
    CREATE (cB0:Chunk {id: $cB0, content: 'B0 — mentions shared + B-only entities'})
    CREATE (eShared:Entity {id: $eShared})
    CREATE (eOnlyA:Entity {id: $eOnlyA})
    CREATE (eOnlyB:Entity {id: $eOnlyB})

    CREATE (docA)-[:HAS_CHUNK {index: 0}]->(cA0)
    CREATE (docA)-[:HAS_CHUNK {index: 1}]->(cA1)
    CREATE (docB)-[:HAS_CHUNK {index: 0}]->(cB0)

    CREATE (cA0)-[:MENTIONS]->(eShared)
    CREATE (cA1)-[:MENTIONS]->(eOnlyA)
    CREATE (cB0)-[:MENTIONS]->(eShared)
    CREATE (cB0)-[:MENTIONS]->(eOnlyB)
    `,
    ID,
  );
}

/** True iff a node with the given id currently exists. */
async function exists(session: GraphSession, id: string): Promise<boolean> {
  const res = await session.run(
    `MATCH (n {id: $id}) RETURN count(n) AS c`,
    { id },
  );
  return res.records[0].get("c").toNumber() > 0;
}

describe("TextFileModule — seeded data", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  beforeEach(() => clear(session));
  after(async () => {
    await clear(session);
    await closeDriver();
  });

  it("hydrates the document with its chunks and mentioned entities", async () => {
    await seed(session);

    const doc = await TextFileModule.fromLazyGraph(session, ID.docA);
    assert.equal(doc.id, ID.docA);

    const chunks = await doc.HAS_CHUNK;
    assert.equal(chunks.length, 2, "docA should have 2 chunks");

    // Chunks come back keyed by id; their content + HAS_CHUNK index survive.
    const byId = new Map(chunks.map((c: any) => [c.id, c]));
    assert.ok(byId.has(ID.cA0) && byId.has(ID.cA1));
    assert.equal(byId.get(ID.cA0).content, "A0 — mentions the shared entity");

    // cA0 mentions exactly the shared entity.
    const c0 = chunks.find((c: any) => c.id === ID.cA0)!;
    const mentioned = await (c0 as any).MENTIONS;
    assert.equal(mentioned.length, 1);
    assert.equal(mentioned[0].id, ID.eShared);
  });

  it("seeds the shared entity with two incoming MENTIONS (one per document)", async () => {
    await seed(session);

    const res = await session.run(
      `MATCH (:Chunk)-[:MENTIONS]->(e:Entity {id: $id}) RETURN count(*) AS c`,
      { id: ID.eShared },
    );
    assert.equal(res.records[0].get("c").toNumber(), 2);
  });
});

describe("TextFileModule.delete", () => {
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
    await TextFileModule.delete(session, ID.docA);
    assert.equal(await exists(session, ID.docA), false, "docA should be gone");
  });

  it("deletes every chunk owned by the document — no orphaned chunks", async () => {
    await TextFileModule.delete(session, ID.docA);

    assert.equal(await exists(session, ID.cA0), false);
    assert.equal(await exists(session, ID.cA1), false);

    // Defensive: no Chunk in our namespace is left without an owning Document.
    const orphans = await session.run(
      `MATCH (c:Chunk) WHERE c.id STARTS WITH $prefix AND NOT ()-[:HAS_CHUNK]->(c)
       RETURN collect(c.id) AS ids`,
      { prefix: PREFIX },
    );
    assert.deepEqual(orphans.records[0].get("ids"), [], "expected no orphaned chunks");
  });

  it("deletes entities that become fully orphaned by the delete", async () => {
    await TextFileModule.delete(session, ID.docA);
    assert.equal(
      await exists(session, ID.eOnlyA),
      false,
      "eOnlyA was mentioned only by docA — it should be deleted",
    );
  });

  it("preserves a shared entity still mentioned by another document", async () => {
    await TextFileModule.delete(session, ID.docA);
    assert.equal(
      await exists(session, ID.eShared),
      true,
      "eShared is still mentioned by docB — it must survive",
    );

    // ...and the surviving MENTIONS edge is exactly docB's (count drops 2 -> 1).
    const res = await session.run(
      `MATCH (c:Chunk)-[:MENTIONS]->(:Entity {id: $id}) RETURN collect(c.id) AS ids`,
      { id: ID.eShared },
    );
    assert.deepEqual(res.records[0].get("ids"), [ID.cB0]);
  });

  it("leaves the other document's subgraph completely intact", async () => {
    await TextFileModule.delete(session, ID.docA);

    assert.equal(await exists(session, ID.docB), true, "docB untouched");
    assert.equal(await exists(session, ID.cB0), true, "cB0 untouched");
    assert.equal(await exists(session, ID.eOnlyB), true, "eOnlyB untouched");

    const doc = await TextFileModule.fromLazyGraph(session, ID.docB);
    const chunks = await doc.HAS_CHUNK;
    assert.equal(chunks.length, 1);
    const mentioned = await (chunks[0] as any).MENTIONS;
    assert.equal(mentioned.length, 2, "cB0 still mentions both shared + B-only entities");
  });

  it("removes the document's MENTIONS / HAS_CHUNK edges entirely", async () => {
    await TextFileModule.delete(session, ID.docA);

    const res = await session.run(
      `MATCH ()-[r]->() WHERE startNode(r).id STARTS WITH 'tf:doc-a'
       RETURN count(r) AS c`,
    );
    assert.equal(res.records[0].get("c").toNumber(), 0, "no edges should originate from docA's deleted nodes");
  });

  it("throws when deleting a document that no longer exists — and changes nothing", async () => {
    await TextFileModule.delete(session, ID.docA);

    // `delete` hydrates the root first, so a second delete rejects (the root
    // node is gone). It is NOT a silent no-op — callers must guard existence.
    await assert.rejects(
      () => TextFileModule.delete(session, ID.docA),
      /No node "Document" with id "tf:doc-a"/,
    );

    // The failed second delete left docB's subgraph (and the shared entity)
    // completely untouched.
    assert.equal(await exists(session, ID.eShared), true);
    assert.equal(await exists(session, ID.docB), true);
    assert.equal(await exists(session, ID.eOnlyB), true);
  });
});
