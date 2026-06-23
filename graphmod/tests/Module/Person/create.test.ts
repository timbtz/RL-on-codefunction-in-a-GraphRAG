// ============================================================
// create.test.ts — PersonService.create
// ============================================================
// Run me:  npm run test:person:create   (or click ▶ Run above that script,
//          or the ▶ gutter icon next to each `it(...)`)
//
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../_db";
import type { GraphSession } from "../../../src/index";
import { PersonModule } from "../../../src/modules/Person/person.module";
import { PersonService } from "../../../src/modules/Person/person.service";

describe("PersonService.create", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  after(closeDriver);

  it("creates a Person node and returns its id", async () => {
    const id = await PersonService.create(session, {
      name: "Ada Lovelace",
      info: "Mathematician — wrote the first algorithm",
    });

    assert.ok(id, "expected a non-empty id");

    const person = await PersonModule.fromLazyGraph(session, id);
    assert.equal(person.id, id);
    assert.equal(person.name, "Ada Lovelace");
    assert.equal(person.info, "Mathematician — wrote the first algorithm");
  });

  it("persists a Person without the optional info property", async () => {
    const id = await PersonService.create(session, { name: "Alan Turing" });

    const person = await PersonModule.fromLazyGraph(session, id);
    assert.equal(person.name, "Alan Turing");
    // Neo4j doesn't store null-valued properties, so an omitted optional
    // hydrates back as `undefined` (the property was never written).
    assert.equal(person.info, undefined);
  });
});
