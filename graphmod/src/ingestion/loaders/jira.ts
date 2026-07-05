// ============================================================
// loaders/jira.ts — parse tickets.json (array of tickets) into RawRecords
// ============================================================
//
// Ticket shape (README):
//   { key, title, description, status, assignee, reporter,
//     components[], references[], blocks[] }
//   id = ticket:<KEY>
//   assignee   -> ASSIGNED person:<handle>
//   reporter   -> REPORTED person:<handle>
//   components -> ABOUT component:<slug>
//   references -> REFERENCES doc:<stem>
//   blocks     -> BLOCKS ticket:<KEY>

import { readFileSync } from "node:fs";
import {
  Resolver,
  docId,
  componentId,
  ticketId,
  type RawRecord,
  type RawRef,
} from "../resolve";

export interface JiraTicket {
  key: string;
  title: string;
  description?: string;
  status?: string;
  assignee?: string;
  reporter?: string;
  components?: string[];
  references?: string[];
  blocks?: string[];
}

/** Parse one ticket object into a RawRecord. */
export function loadJiraTicket(t: JiraTicket, resolver: Resolver): RawRecord {
  const id = ticketId(t.key);
  const refs: RawRef[] = [];

  if (t.assignee) refs.push(resolver.personRef("ASSIGNED", t.assignee));
  if (t.reporter) refs.push(resolver.personRef("REPORTED", t.reporter));
  for (const c of t.components ?? []) {
    refs.push({ rel: "ABOUT", targetId: componentId(c), name: c });
  }
  for (const r of t.references ?? []) {
    refs.push({ rel: "REFERENCES", targetId: docId(r), name: r.replace(/^doc:/, "") });
  }
  for (const b of t.blocks ?? []) {
    refs.push({ rel: "BLOCKS", targetId: ticketId(b), name: b });
  }

  const props: Record<string, unknown> = { id, key: t.key, title: t.title };
  if (t.description != null) props.description = t.description;
  if (t.status != null) props.status = t.status;

  return {
    kind: "jira",
    id,
    label: "Ticket",
    props,
    refs,
    text: `${t.title}\n${t.description ?? ""}`,
    // Description prose becomes body chunks 1..n (mirrors Document bodies), so
    // it is visible to the corpus_ft fulltext index, not just a node prop.
    body: t.description,
  };
}

/** Parse the raw tickets.json contents into RawRecords. */
export function loadJiraJson(raw: string, resolver: Resolver): RawRecord[] {
  const arr = JSON.parse(raw) as JiraTicket[];
  if (!Array.isArray(arr)) throw new Error("tickets.json: expected a top-level array");
  return arr.map((t) => loadJiraTicket(t, resolver));
}

/** Load tickets.json from disk. */
export function loadJiraFile(filePath: string, resolver: Resolver): RawRecord[] {
  return loadJiraJson(readFileSync(filePath, "utf8"), resolver);
}
