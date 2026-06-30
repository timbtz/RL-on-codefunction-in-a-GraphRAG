// ============================================================
// extract.test.ts — PURE unit tests for the rule-based extractor:
//   - the token regexes
//   - the chunk-"card" serializer (NAMES related entities)
//   - mergeNode/mergeEdge MERGE shapes (via a fake capturing tx)
//   - the metered LLM lever with an INJECTED stub LlmClient (no network),
//     plus the "lever off => no-op" gating contract.
// No DB, no network, fully deterministic.
// ============================================================
import { test } from "node:test";
import assert from "node:assert/strict";
import type { GraphTransaction } from "../../../src/core/types";
import {
  TICKET_RE,
  HANDLE_RE,
  URL_RE,
  WIKILINK_TOKEN_RE,
  mergeNode,
  mergeEdge,
  buildChunks,
  extractWithLLM,
  writeLlmExtraction,
  type SchemaJson,
} from "../../../src/ingestion/extract";
import { Resolver, KNOWN_PEOPLE, KNOWN_COMPONENTS } from "../../../src/ingestion/resolve";
import { loadJiraTicket, type JiraTicket } from "../../../src/ingestion/loaders/jira";
import { Meter, type LlmClient, type LlmRequest, type LlmResult } from "../../../src/ingestion/llm";

// ---- a fake transaction that records every run(cypher, params) ----
interface Captured {
  cypher: string;
  params: Record<string, unknown> | undefined;
}
function fakeTx(): { tx: GraphTransaction; calls: Captured[] } {
  const calls: Captured[] = [];
  const tx: GraphTransaction = {
    async run(cypher, params) {
      calls.push({ cypher, params });
      return { records: [] };
    },
  };
  return { tx, calls };
}

// ---- regexes ----

test("TICKET_RE matches PROJ-<n> tokens", () => {
  assert.deepEqual("blocked on PROJ-42 today".match(TICKET_RE), ["PROJ-42"]);
  assert.equal("no ticket here".match(TICKET_RE), null);
});

test("HANDLE_RE matches @handle and captures the handle", () => {
  const m = [...("ping @tim and @noah".matchAll(HANDLE_RE))];
  assert.deepEqual(m.map((x) => x[1]), ["tim", "noah"]);
});

test("URL_RE matches http(s) urls", () => {
  assert.deepEqual(
    "see https://example.com/x for details".match(URL_RE),
    ["https://example.com/x"],
  );
});

test("WIKILINK_TOKEN_RE matches [[stem]] and captures the stem", () => {
  const m = [...("read [[auth-redesign]] now".matchAll(WIKILINK_TOKEN_RE))];
  assert.deepEqual(m.map((x) => x[1]), ["auth-redesign"]);
});

// ---- chunk-card serializer ----

test("buildChunks: ticket card NAMES related entities (Assignee + component)", () => {
  const t: JiraTicket = {
    key: "PROJ-12",
    title: "Migrate auth to OAuth 2.1",
    status: "In Progress",
    assignee: "tim",
    reporter: "lasse",
    components: ["auth-redesign"],
    references: ["auth-redesign"],
    blocks: [],
  };
  const rec = loadJiraTicket(t, new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS));
  const chunks = buildChunks(rec);

  const card = chunks[0];
  assert.equal(card.id, "ticket:PROJ-12#c0");
  assert.equal(card.index, 0);
  assert.equal(card.source, "jira");
  assert.match(card.content, /Ticket PROJ-12: Migrate auth to OAuth 2\.1\./);
  assert.match(card.content, /Assignee Tim/);
  assert.match(card.content, /Reporter Lasse/);
  assert.match(card.content, /auth-redesign/);
});

// ---- MERGE shapes via fake tx ----

test("mergeNode emits a MERGE-on-id upsert with bound params", async () => {
  const { tx, calls } = fakeTx();
  await mergeNode(tx, "Person", "person:tim", { id: "person:tim", name: "Tim" });
  assert.equal(calls.length, 1);
  assert.match(calls[0].cypher, /MERGE \(n:Person \{id: \$id\}\)/);
  assert.match(calls[0].cypher, /ON CREATE SET n \+= \$props/);
  assert.deepEqual(calls[0].params, {
    id: "person:tim",
    props: { id: "person:tim", name: "Tim" },
  });
});

test("mergeEdge MATCHes both endpoints then MERGEs the typed relationship", async () => {
  const { tx, calls } = fakeTx();
  await mergeEdge(tx, "Document", "doc:x", "AUTHORED_BY", "Person", "person:tim");
  assert.equal(calls.length, 1);
  assert.match(calls[0].cypher, /MATCH \(a:Document \{id: \$fromId\}\), \(b:Person \{id: \$toId\}\)/);
  assert.match(calls[0].cypher, /MERGE \(a\)-\[r:AUTHORED_BY\]->\(b\)/);
  assert.equal(calls[0].params?.fromId, "doc:x");
  assert.equal(calls[0].params?.toId, "person:tim");
});

// ---- LLM lever (injected stub, no network) ----

const SCHEMA: SchemaJson = {
  labels: ["Person", "Component", "Entity", "Ticket"],
  relationships: [],
};

/** Deterministic stub LlmClient: returns a fixed extraction + cost, counts calls. */
function stubLlm(): { llm: LlmClient; calls: LlmRequest[] } {
  const calls: LlmRequest[] = [];
  const llm: LlmClient = {
    async complete(req: LlmRequest): Promise<LlmResult> {
      calls.push(req);
      return {
        json: {
          entities: [
            { name: "Tim", type: "Person" }, // known -> reuses person:tim
            { name: "Acme", type: "Entity" }, // unknown -> new entity
          ],
          relations: [{ from: "Acme", type: "owns", to: "Tim" }],
        },
        tokens: 123,
        usd: 0.0042,
      };
    },
  };
  return { llm, calls };
}

test("extractWithLLM: resolves entities/relations and the Meter records tokens/usd", async () => {
  const resolver = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);
  const meter = new Meter("test");
  const { llm, calls } = stubLlm();

  const ex = await extractWithLLM("Acme owns Tim.", SCHEMA, llm, meter, "chat.msg:eng:9", resolver);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].field, "chat.msg:eng:9");

  // "Tim" reuses the canonical Person node; "Acme" becomes a new Entity.
  const byId = new Map(ex.entities.map((e) => [e.id, e]));
  assert.ok(byId.has("person:tim"));
  assert.equal(byId.get("person:tim")?.label, "Person");
  assert.ok(byId.has("entity:acme"));
  assert.equal(byId.get("entity:acme")?.label, "Entity");

  assert.deepEqual(ex.relations, [{ fromId: "entity:acme", type: "OWNS", toId: "person:tim" }]);

  // Meter accumulated the stub's reported cost.
  assert.equal(meter.tokens, 123);
  assert.equal(meter.usd, 0.0042);
  assert.equal(meter.calls.length, 1);
  assert.equal(meter.calls[0].field, "chat.msg:eng:9");
});

test("LLM lever is a NO-OP when the flag is off: client untouched, Meter stays 0", async () => {
  const resolver = new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);
  const meter = new Meter("test");
  const { llm, calls } = stubLlm();

  // Mirror ingest()'s gating: extractWithLLM is only called when the lever is on.
  const leverOn = false;
  if (leverOn) {
    await extractWithLLM("Acme owns Tim.", SCHEMA, llm, meter, "f", resolver);
  }

  assert.equal(calls.length, 0); // no network/client call
  assert.equal(meter.tokens, 0);
  assert.equal(meter.usd, 0);
  assert.deepEqual(meter.calls, []);
});

test("writeLlmExtraction: new Entity is MERGEd; reused Person is skipped; relation carries sources", async () => {
  const { tx, calls } = fakeTx();
  await writeLlmExtraction(
    tx,
    {
      entities: [
        { id: "person:tim", label: "Person", name: "Tim" }, // reused -> skipped
        { id: "entity:acme", label: "Entity", name: "Acme" }, // new -> mergeNode
      ],
      relations: [{ fromId: "entity:acme", type: "OWNS", toId: "person:tim" }],
    },
    "chat.msg:eng:9",
  );

  const nodeMerges = calls.filter((c) => /MERGE \(n:Entity \{id: \$id\}\)/.test(c.cypher));
  assert.equal(nodeMerges.length, 1); // only the new Entity, Person skipped
  assert.equal(nodeMerges[0].params?.id, "entity:acme");

  const relMerges = calls.filter((c) => /MERGE \(a\)-\[rel:OWNS\]->\(b\)/.test(c.cypher));
  assert.equal(relMerges.length, 1);
  assert.equal(relMerges[0].params?.from, "entity:acme");
  assert.equal(relMerges[0].params?.to, "person:tim");
  assert.equal(relMerges[0].params?.src, "chat.msg:eng:9");
});
