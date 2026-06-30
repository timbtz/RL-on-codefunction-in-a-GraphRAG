// ============================================================
// loaders.test.ts — PURE unit tests for the markdown / chat / jira loaders.
// Feeds inline minimal fixtures (mirrors the real corpus) through each loader
// and asserts the RawRecord id/label/props and the resolved RawRef edges.
// No DB, no fs, no network.
// ============================================================
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  Resolver,
  KNOWN_PEOPLE,
  KNOWN_COMPONENTS,
  type RawRecord,
  type RawRef,
  type RelType,
} from "../../../src/ingestion/resolve";
import { loadMarkdownDoc, parseFrontmatter } from "../../../src/ingestion/loaders/markdown";
import { loadChatMessage, type ChatMessage } from "../../../src/ingestion/loaders/chat";
import { loadJiraTicket, type JiraTicket } from "../../../src/ingestion/loaders/jira";

const resolver = () => new Resolver(KNOWN_PEOPLE, KNOWN_COMPONENTS);

/** Find the (rel,targetId) pairs for one relationship type. */
function refsFor(rec: RawRecord, rel: RelType): RawRef[] {
  return rec.refs.filter((r) => r.rel === rel);
}

// ---- markdown ----

test("parseFrontmatter splits scalar + list fields from body", () => {
  const raw = [
    "---",
    "id: doc:auth-redesign",
    "title: Auth Redesign Spec",
    "author: tim",
    "components: [auth-redesign]",
    "---",
    "Body text here with a [[search-index-design]] link.",
  ].join("\n");
  const { fields, body } = parseFrontmatter(raw);
  assert.equal(fields.id, "doc:auth-redesign");
  assert.equal(fields.title, "Auth Redesign Spec");
  assert.equal(fields.author, "tim");
  assert.deepEqual(fields.components, ["auth-redesign"]);
  assert.match(body, /search-index-design/);
});

test("loadMarkdownDoc: author->AUTHORED_BY, components->ABOUT, [[wikilink]]->REFERENCES", () => {
  const raw = [
    "---",
    "id: doc:auth-redesign",
    "title: Auth Redesign Spec",
    "author: tim",
    "components: [auth-redesign]",
    "---",
    "Spec body. See the [[search-index-design]] path.",
  ].join("\n");
  const rec = loadMarkdownDoc("/corpus/docs/auth-redesign.md", raw, resolver());

  assert.equal(rec.kind, "doc");
  assert.equal(rec.id, "doc:auth-redesign");
  assert.equal(rec.label, "Document");
  assert.equal(rec.props.id, "doc:auth-redesign");
  assert.equal(rec.props.title, "Auth Redesign Spec");
  assert.equal(rec.props.path, "/corpus/docs/auth-redesign.md");

  assert.deepEqual(refsFor(rec, "AUTHORED_BY"), [
    { rel: "AUTHORED_BY", targetId: "person:tim", name: "Tim" },
  ]);
  assert.deepEqual(refsFor(rec, "ABOUT"), [
    { rel: "ABOUT", targetId: "component:auth-redesign", name: "auth-redesign" },
  ]);
  assert.deepEqual(refsFor(rec, "REFERENCES"), [
    { rel: "REFERENCES", targetId: "doc:search-index-design", name: "search-index-design" },
  ]);
});

test("loadMarkdownDoc: id falls back to filename stem when frontmatter omits it", () => {
  const raw = "---\ntitle: Onboarding\nauthor: noah\n---\nNo id field here.";
  const rec = loadMarkdownDoc("/corpus/docs/onboarding.md", raw, resolver());
  assert.equal(rec.id, "doc:onboarding");
});

// ---- chat ----

test("loadChatMessage: author->SENT_BY, reply_to->REPLY_TO, [[wikilink]]->REFERENCES", () => {
  const m: ChatMessage = {
    seq: 2,
    channel: "eng",
    author: "lasse",
    ts: "2026-05-01T10:15:00Z",
    reply_to: 1,
    text: "@tim the [[auth-redesign]] looks good - approving it.",
  };
  const rec = loadChatMessage(m, resolver());

  assert.equal(rec.kind, "chat");
  assert.equal(rec.id, "msg:eng:2");
  assert.equal(rec.label, "Message");
  assert.equal(rec.props.id, "msg:eng:2");
  assert.equal(rec.props.channel, "eng");
  assert.equal(rec.props.text, m.text);

  assert.deepEqual(refsFor(rec, "SENT_BY"), [
    { rel: "SENT_BY", targetId: "person:lasse", name: "Lasse" },
  ]);
  assert.deepEqual(refsFor(rec, "REPLY_TO"), [
    { rel: "REPLY_TO", targetId: "msg:eng:1", name: "message 1" },
  ]);
  assert.deepEqual(refsFor(rec, "REFERENCES"), [
    { rel: "REFERENCES", targetId: "doc:auth-redesign", name: "auth-redesign" },
  ]);
});

test("loadChatMessage: explicit references[] become REFERENCES doc edges", () => {
  const m: ChatMessage = {
    seq: 4,
    channel: "eng",
    author: "noah",
    text: "New hires should read this.",
    references: ["billing-overview"],
  };
  const rec = loadChatMessage(m, resolver());
  assert.deepEqual(refsFor(rec, "REFERENCES"), [
    { rel: "REFERENCES", targetId: "doc:billing-overview", name: "billing-overview" },
  ]);
});

// ---- jira ----

test("loadJiraTicket: assignee/reporter/components/references/blocks resolve to canonical ids", () => {
  const t: JiraTicket = {
    key: "PROJ-19",
    title: "Auth incident hardening",
    description: "Harden the token cache eviction policy.",
    status: "Open",
    assignee: "tim",
    reporter: "mara",
    components: ["auth-redesign"],
    references: ["incident-2026-05"],
    blocks: ["PROJ-7"],
  };
  const rec = loadJiraTicket(t, resolver());

  assert.equal(rec.kind, "jira");
  assert.equal(rec.id, "ticket:PROJ-19");
  assert.equal(rec.label, "Ticket");
  assert.equal(rec.props.key, "PROJ-19");
  assert.equal(rec.props.title, "Auth incident hardening");
  assert.equal(rec.props.status, "Open");

  assert.deepEqual(refsFor(rec, "ASSIGNED"), [
    { rel: "ASSIGNED", targetId: "person:tim", name: "Tim" },
  ]);
  assert.deepEqual(refsFor(rec, "REPORTED"), [
    { rel: "REPORTED", targetId: "person:mara", name: "Mara" },
  ]);
  assert.deepEqual(refsFor(rec, "ABOUT"), [
    { rel: "ABOUT", targetId: "component:auth-redesign", name: "auth-redesign" },
  ]);
  assert.deepEqual(refsFor(rec, "REFERENCES"), [
    { rel: "REFERENCES", targetId: "doc:incident-2026-05", name: "incident-2026-05" },
  ]);
  assert.deepEqual(refsFor(rec, "BLOCKS"), [
    { rel: "BLOCKS", targetId: "ticket:PROJ-7", name: "PROJ-7" },
  ]);
});
