// ============================================================
// idempotency.test.ts — DB-GATED, DEFERRED Level-2 integration test.
//
// REQUIRES A LIVE NEO4J. This file is NOT runnable in the authoring sandbox:
// it needs a reachable Neo4j instance configured via the standard env vars
//   NEO4J_URI / NEO4J_USER / NEO4J_PASS / NEO4J_DB   (see tests/_db.ts; loaded
// from .env via `dotenv/config`). It is authored for a DEFERRED run on the
// Level-2 harness where a database is available. It still type-checks/compiles
// without a DB; it simply must not be executed here.
//
// Contract under test: `ingest()` uses idempotent MERGE-upserts, so ingesting
// the SAME corpus twice converges to the SAME graph (no duplicate nodes/rels).
// ============================================================
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { openSession, closeDriver } from "../../_db";
import type { GraphSession } from "../../../src/index";
import { ingest } from "../../../src/ingestion/ingest";

// This file: graphmod/tests/Module/Corpus/idempotency.test.ts
// Corpus:    graphsearch/data/corpus  (worktree-root sibling of graphmod/)
const __dirname = dirname(fileURLToPath(import.meta.url));
const CORPUS_DIR = join(__dirname, "../../../../graphsearch/data/corpus");

async function counts(session: GraphSession): Promise<{ nodes: number; rels: number }> {
  const toInt = (v: any): number =>
    typeof v === "number" ? v : v && typeof v === "object" && "low" in v ? v.low : Number(v ?? 0);
  const n = await session.run("MATCH (n) RETURN count(n) AS c");
  const r = await session.run("MATCH ()-[x]->() RETURN count(x) AS c");
  return { nodes: toInt(n.records[0].get("c")), rels: toInt(r.records[0].get("c")) };
}

describe("ingest idempotency (DB-gated, deferred Level-2)", () => {
  let session: GraphSession;
  before(async () => {
    session = openSession();
    // Start from a clean graph so the counts are meaningful + deterministic.
    await session.run("MATCH (n) DETACH DELETE n");
  });
  after(closeDriver);

  it("re-ingesting the same corpus does not change node/relationship counts", async () => {
    // LLM lever stays OFF (default) -> no network, deterministic rule-based build.
    await ingest(session, CORPUS_DIR);
    const first = await counts(session);

    assert.ok(first.nodes > 0, "expected the first ingest to create nodes");
    assert.ok(first.rels > 0, "expected the first ingest to create relationships");

    await ingest(session, CORPUS_DIR);
    const second = await counts(session);

    assert.equal(second.nodes, first.nodes, "node count must be stable on re-ingest");
    assert.equal(second.rels, first.rels, "relationship count must be stable on re-ingest");
  });
});
