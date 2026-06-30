// ============================================================
// loaders/markdown.ts — parse docs/*.md (YAML-ish frontmatter + body)
// ============================================================
//
// Frontmatter shape (README):
//   ---
//   id: doc:<stem>
//   title: <string>
//   author: <handle>        -> AUTHORED_BY person:<handle>
//   components: [<slug>...]  -> ABOUT component:<slug>
//   ---
//   <body>                   -> Chunk(s); [[wikilink]] -> REFERENCES doc:<stem>
//
// No YAML dependency (neo4j-driver is the only runtime dep), so the small,
// fixed frontmatter grammar is hand-parsed.

import { readFileSync, readdirSync } from "node:fs";
import { join, basename } from "node:path";
import {
  Resolver,
  docId,
  componentId,
  type RawRecord,
  type RawRef,
} from "../resolve";

/** Wikilinks `[[stem]]` anywhere in a string. */
export const WIKILINK_RE = /\[\[([^\]]+)\]\]/g;

interface Frontmatter {
  fields: Record<string, string | string[]>;
  body: string;
}

/** Split `--- ... ---` frontmatter from the markdown body and parse scalar/list fields. */
export function parseFrontmatter(raw: string): Frontmatter {
  const fields: Record<string, string | string[]> = {};
  const text = raw.replace(/^﻿/, ""); // strip a leading UTF-8 BOM
  const m = text.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?([\s\S]*)$/);
  if (!m) return { fields, body: text.trim() };

  for (const line of m[1].split(/\r?\n/)) {
    const kv = line.match(/^([A-Za-z0-9_-]+)\s*:\s*(.*)$/);
    if (!kv) continue;
    const key = kv[1].trim();
    const valRaw = kv[2].trim();
    if (valRaw.startsWith("[") && valRaw.endsWith("]")) {
      const inner = valRaw.slice(1, -1).trim();
      fields[key] = inner
        ? inner.split(",").map((s) => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean)
        : [];
    } else {
      fields[key] = valRaw.replace(/^["']|["']$/g, "");
    }
  }
  return { fields, body: m[2].trim() };
}

/** Parse a single markdown document into a RawRecord. */
export function loadMarkdownDoc(
  filePath: string,
  raw: string,
  resolver: Resolver,
): RawRecord {
  const { fields, body } = parseFrontmatter(raw);
  const stem = basename(filePath).replace(/\.md$/i, "");
  const id =
    typeof fields.id === "string" && fields.id ? docId(fields.id) : docId(stem);
  const title = (typeof fields.title === "string" && fields.title) || stem;

  const refs: RawRef[] = [];

  // author -> AUTHORED_BY
  if (typeof fields.author === "string" && fields.author) {
    refs.push(resolver.personRef("AUTHORED_BY", fields.author));
  }
  // components -> ABOUT
  const components = Array.isArray(fields.components) ? fields.components : [];
  for (const c of components) {
    refs.push({ rel: "ABOUT", targetId: componentId(c), name: c });
  }
  // [[wikilink]] in body -> REFERENCES doc
  const seenDocRefs = new Set<string>();
  for (const wm of body.matchAll(WIKILINK_RE)) {
    const stemRef = wm[1].trim();
    const tid = docId(stemRef);
    if (tid === id || seenDocRefs.has(tid)) continue;
    seenDocRefs.add(tid);
    refs.push({ rel: "REFERENCES", targetId: tid, name: stemRef });
  }

  return {
    kind: "doc",
    id,
    label: "Document",
    props: { id, title, path: filePath },
    refs,
    text: `${title}\n${body}`,
    body,
  };
}

/** Load every `*.md` under `docsDir` into RawRecords. */
export function loadMarkdownDir(docsDir: string, resolver: Resolver): RawRecord[] {
  const files = readdirSync(docsDir)
    .filter((f) => /\.md$/i.test(f))
    .sort();
  return files.map((f) => {
    const fp = join(docsDir, f);
    return loadMarkdownDoc(fp, readFileSync(fp, "utf8"), resolver);
  });
}
