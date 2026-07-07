// ============================================================
// schema.test.ts — PURE unit tests for the Corpus modules' declared vocabulary
// (what export-schema.ts unions into schema.json). Guards that every MENTIONS
// endpoint combination the extractor actually writes (Chunk -> Entity / Person /
// Component / Ticket, see collectMentions) is declared somewhere in the modules,
// and that declared prop types match what the loaders store.
// No DB, no network, fully deterministic.
// ============================================================
import { test } from "node:test";
import assert from "node:assert/strict";
import { CORPUS_MODULES, MessageModule } from "../../../src/modules/Corpus/index";
import type { NodeDef, RelationshipDef } from "../../../src/core/types";

/** Union all plain-relationship triplets as "TYPE:from->to" (mirrors export-schema). */
function declaredTriplets(): Set<string> {
  const out = new Set<string>();
  for (const mod of CORPUS_MODULES) {
    for (const [type, rel] of Object.entries(
      mod.schema.relationships as Record<string, unknown>,
    )) {
      const r = rel as RelationshipDef;
      if (typeof r?.from !== "string" || typeof r?.to !== "string") continue;
      out.add(`${type}:${r.from}->${r.to}`);
    }
  }
  return out;
}

test("modules declare every MENTIONS triplet the extractor writes", () => {
  const triplets = declaredTriplets();
  for (const target of ["Entity", "Person", "Component", "Ticket"]) {
    assert.ok(
      triplets.has(`MENTIONS:Chunk->${target}`),
      `missing MENTIONS Chunk->${target} declaration`,
    );
  }
});

test("Message.ts is declared as the ISO string the chat loader stores", () => {
  const node = (MessageModule.schema.nodes as Record<string, NodeDef>).Message;
  assert.equal(node.properties.ts, "string?");
});
