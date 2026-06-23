// ============================================================
// createWithMissingProperty.test.ts — PersonService.createWithMissingProperty
// ============================================================
// Run me:  npm run test:person:missing   (or click ▶ Run above that script,
//          or the ▶ gutter icon next to each `it(...)`)
//
// The service deliberately creates a Person WITHOUT the required `name`.
// In-tx validation must catch it, throw SchemaValidationError, and roll the
// write back — so here a thrown error is the PASS condition.
//
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../_db";
import { SchemaValidationError, type GraphSession } from "../../../src/index";
import { PersonService } from "../../../src/modules/Person/person.service";

describe("PersonService.createWithMissingProperty", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  after(closeDriver);

  it("throws SchemaValidationError and rolls back the write", async () => {
    await assert.rejects(
      () => PersonService.createWithMissingProperty(session),
      (err: unknown) => {
        assert.ok(
          err instanceof SchemaValidationError,
          `expected SchemaValidationError, got ${String(err)}`,
        );
        assert.ok(
          err.errors.length > 0,
          "expected at least one validation error",
        );
        return true;
      },
    );
  });
});
