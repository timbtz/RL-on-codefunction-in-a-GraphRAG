// ============================================================
// loaders/github.ts — parse github.json (pull requests) into RawRecords
// ============================================================
//
// PR shape:
//   { repo, number, title, description?, state?, author?, reviewers[]?,
//     fixes[]?, components[]?, references[]? }
//   id = pr:<repo>#<number>
//   author     -> AUTHORED_BY person:<handle>
//   reviewers  -> REVIEWED_BY person:<handle>
//   fixes      -> FIXES ticket:<KEY>        (the PR that resolves the Jira ticket)
//   components -> ABOUT component:<slug>
//   references -> REFERENCES doc:<stem>
//   PROJ-\d+ in title/description -> MENTIONS (handled in extract.ts)

import { readFileSync } from "node:fs";
import {
  Resolver,
  docId,
  componentId,
  ticketId,
  prId,
  type RawRecord,
  type RawRef,
} from "../resolve";

export interface GithubPr {
  repo: string;
  number: number;
  title: string;
  description?: string;
  state?: string;
  author?: string;
  reviewers?: string[];
  fixes?: string[];
  components?: string[];
  references?: string[];
}

/** Parse one PR object into a RawRecord. */
export function loadPr(p: GithubPr, resolver: Resolver): RawRecord {
  const id = prId(p.repo, p.number);
  const refs: RawRef[] = [];

  if (p.author) refs.push(resolver.personRef("AUTHORED_BY", p.author));
  for (const r of p.reviewers ?? []) refs.push(resolver.personRef("REVIEWED_BY", r));
  for (const t of p.fixes ?? []) {
    refs.push({ rel: "FIXES", targetId: ticketId(t), name: t.replace(/^ticket:/, "") });
  }
  for (const c of p.components ?? []) {
    refs.push({ rel: "ABOUT", targetId: componentId(c), name: c });
  }
  for (const r of p.references ?? []) {
    refs.push({ rel: "REFERENCES", targetId: docId(r), name: r.replace(/^doc:/, "") });
  }

  const props: Record<string, unknown> = {
    id,
    number: p.number,
    title: p.title,
    repo: p.repo,
  };
  if (p.state != null) props.state = p.state;
  if (p.description != null) props.description = p.description;

  return {
    kind: "github",
    id,
    label: "PullRequest",
    props,
    refs,
    text: `${p.title}\n${p.description ?? ""}`,
  };
}

/** Parse the raw github.json contents into RawRecords. */
export function loadGithubJson(raw: string, resolver: Resolver): RawRecord[] {
  const arr = JSON.parse(raw) as GithubPr[];
  if (!Array.isArray(arr)) throw new Error("github.json: expected a top-level array");
  return arr.map((p) => loadPr(p, resolver));
}

/** Load github.json from disk. */
export function loadGithubFile(filePath: string, resolver: Resolver): RawRecord[] {
  return loadGithubJson(readFileSync(filePath, "utf8"), resolver);
}
