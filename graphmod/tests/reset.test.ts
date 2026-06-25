// ============================================================
// reset.test.ts — reset the database (delete all nodes)
// ============================================================
// Run me:  npm run test:reset   (or click ▶ Run above the `it(...)`,
//          or the ▶ gutter icon next to it)
//
// ⚠️  DESTRUCTIVE. Deletes EVERY node and relationship in the target database,
//     leaving an empty graph. Point it at a dev database only (connection
//     comes from .env).
//
import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "./_db";

describe("reset", () => {
  after(closeDriver);

  it("deletes every node and relationship", async () => {
    const session = openSession();
    await session.run(`MATCH (n) DETACH DELETE n`);

    const result = await session.run(`MATCH (n) RETURN count(n) AS count`);
    const remaining = result.records[0].get("count").toNumber();
    assert.equal(remaining, 0, "expected an empty graph after reset");

    console.log("✅ Database reset — all nodes and relationships deleted.");
  });
});
