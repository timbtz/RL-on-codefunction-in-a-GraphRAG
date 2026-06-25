// ============================================================
// linkAccountToPerson.test.ts — MicrosoftAccountService
// ============================================================
// Run me:  npx tsx --test -r dotenv/config tests/Microsoft/*.test.ts
//          (or the ▶ gutter icon next to each `it(...)`)
//
// Graph shape under test:
//
//   (:MicrosoftAccount)-[:ACCOUNT_OF]->(:Person)     // cardinality "1"
//
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openSession, closeDriver } from "../../_db";
import type { GraphSession } from "../../../src/index";
import { PersonService } from "../../../src/modules/Person/person.service";
import { MicrosoftAccountService } from "../../../src/modules/Microsoft/MicrosoftAccount.service";
import { SchemaValidationError } from "../../../src/index";
import { MicrosoftAccountModule } from "../../../src/modules/Microsoft/MicrosoftAccount.module";


describe("MicrosoftAccountService", () => {
  let session: GraphSession;
  before(() => {
    session = openSession();
  });
  after(closeDriver);

  // ---- test1: create an account ----
  it("creates a dummy account and returns its id", async () => {
    const accountId = await MicrosoftAccountService._createDummyAccount(session);

    assert.ok(accountId, "expected a non-empty account id");

    const result = await session.run(
      `MATCH (a:MicrosoftAccount { id: $accountId })
       RETURN a.id AS id, a.userPrincipalName AS upn, a.accountEnabled AS enabled`,
      { accountId },
    );
    assert.equal(result.records.length, 1, "expected the account to exist");
    assert.equal(result.records[0].get("id"), accountId);
    assert.ok(result.records[0].get("upn"), "expected a userPrincipalName");

    const account = await MicrosoftAccountModule.fromLazyGraph(session, accountId);
    // A standalone account has no ACCOUNT_OF edge yet.
    assert.equal((await account.ACCOUNT_OF), null);
  });

  // ---- test2: create account, create user, link user to account ----
  it("links an account to a single person via ACCOUNT_OF", async () => {
    const accountId = await MicrosoftAccountService._createDummyAccount(session);
    const personId = await PersonService.create(session, { name: "Grace Hopper" });

    await MicrosoftAccountService.linkAccountToPerson(session, accountId, personId);

    const account = await MicrosoftAccountModule.fromLazyGraph(session, accountId);

    assert.equal(await account.ACCOUNT_OF.id, personId);
  });

  // ---- test3: create account, create two users, link account to two users ----
  it("rejects linking one account to a second person (cardinality \"1\")", async () => {
    const accountId = await MicrosoftAccountService._createDummyAccount(session);
    const personA = await PersonService.create(session, { name: "Edsger Dijkstra" });
    const personB = await PersonService.create(session, { name: "Donald Knuth" });

    // First link succeeds — the account now belongs to exactly one Person.
    await MicrosoftAccountService.linkAccountToPerson(session, accountId, personA);

    const account = await MicrosoftAccountModule.fromLazyGraph(session, accountId);
    
    assert.equal(await account.ACCOUNT_OF.id, personA);

    // ACCOUNT_OF is cardinality "1", so a second link pushes the count to 2 and
    // fails in-tx validation, rolling back the second edge.
    try {
      await MicrosoftAccountService.linkAccountToPerson(session, accountId, personB);
    } catch (error) {
      assert.ok(error instanceof SchemaValidationError, "expected a SchemaValidationError when linking a second account to a person");
    }

    // The rollback means the account is still linked to only the first Person.
    assert.equal(await account.ACCOUNT_OF.id, personA);
  });
});
