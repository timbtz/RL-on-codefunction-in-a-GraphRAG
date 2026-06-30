// ============================================================
// resolve.test.ts — PURE unit tests for canonical-id helpers + gazetteer.
// No DB, no network, fully deterministic.
// ============================================================
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  slug,
  personId,
  componentId,
  entityId,
  docId,
  msgId,
  ticketId,
  chunkId,
  Resolver,
  KNOWN_PEOPLE,
  KNOWN_COMPONENTS,
} from "../../../src/ingestion/resolve";

test("slug: lowercases, maps non-alnum runs to '-', collapses repeats, trims", () => {
  assert.equal(slug("Auth Redesign"), "auth-redesign");
  assert.equal(slug("  Hello,  World!! "), "hello-world");
  assert.equal(slug("PROJ-12"), "proj-12");
  assert.equal(slug("___multiple---separators___"), "multiple-separators");
  assert.equal(slug("Tim"), "tim");
});

test("*Id helpers produce CONTRACT-shaped canonical ids", () => {
  assert.equal(personId("tim"), "person:tim");
  assert.equal(personId("Tim"), "person:tim"); // slugged
  assert.equal(componentId("auth-redesign"), "component:auth-redesign");
  assert.equal(componentId("Search Index"), "component:search-index");
  assert.equal(entityId("Some Thing"), "entity:some-thing");
  // Ticket KEY kept verbatim (uppercase), NOT slugged.
  assert.equal(ticketId("PROJ-12"), "ticket:PROJ-12");
  assert.equal(ticketId("ticket:PROJ-7"), "ticket:PROJ-7"); // idempotent on prefixed input
  // Document stem.
  assert.equal(docId("auth-redesign"), "doc:auth-redesign");
  assert.equal(docId("doc:auth-redesign"), "doc:auth-redesign"); // idempotent on prefixed input
  // Message channel:seq.
  assert.equal(msgId("eng", 2), "msg:eng:2");
  assert.equal(msgId("eng", "10"), "msg:eng:10");
  // Chunk owner#cIndex.
  assert.equal(chunkId("doc:auth-redesign", 0), "doc:auth-redesign#c0");
  assert.equal(chunkId(ticketId("PROJ-12"), 3), "ticket:PROJ-12#c3");
});

test("Resolver gazetteer matches known person name + alias case-insensitively", () => {
  const r = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);

  // Display name surface form.
  const byName = r.gazetteerMentions("Spoke to Tim about the rollout");
  assert.deepEqual(
    byName.map((h) => h.id),
    ["person:tim"],
  );
  assert.equal(byName[0].label, "Person");
  assert.equal(byName[0].name, "Tim");

  // @handle surface form, case-insensitive.
  const byHandle = r.gazetteerMentions("ping @TIM please");
  assert.deepEqual(
    byHandle.map((h) => h.id),
    ["person:tim"],
  );

  // Component slug surface form.
  const byComp = r.gazetteerMentions("this touches billing only");
  assert.deepEqual(
    byComp.map((h) => h.id),
    ["component:billing"],
  );
});

test("Resolver gazetteer returns no hit for an unknown token", () => {
  const r = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);
  assert.deepEqual(r.gazetteerMentions("an unrelated zoltan sentence"), []);
});

test("Resolver.personName: known handle -> vocab name, unknown -> capitalized", () => {
  const r = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);
  assert.equal(r.personName("tim"), "Tim");
  assert.equal(r.personName("Tim"), "Tim");
  assert.equal(r.personName("jane-doe"), "Jane Doe");
});

test("Resolver.personRef builds a resolved AUTHORED_BY-style ref", () => {
  const r = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);
  assert.deepEqual(r.personRef("SENT_BY", "tim"), {
    rel: "SENT_BY",
    targetId: "person:tim",
    name: "Tim",
  });
});
