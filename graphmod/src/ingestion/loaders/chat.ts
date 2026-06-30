// ============================================================
// loaders/chat.ts — parse chat.json (array of messages) into RawRecords
// ============================================================
//
// Message shape (README):
//   { seq, channel, author, ts, text, reply_to?, references?[] }
//   id = msg:<channel>:<seq>
//   author     -> SENT_BY person:<handle>
//   reply_to   -> REPLY_TO msg:<channel>:<reply_to> (same channel)
//   [[wikilink]] / references -> REFERENCES doc:<stem>
//   @handle / PROJ-\d+ / component names -> MENTIONS (handled in extract.ts)

import { readFileSync } from "node:fs";
import {
  Resolver,
  docId,
  msgId,
  type RawRecord,
  type RawRef,
} from "../resolve";
import { WIKILINK_RE } from "./markdown";

export interface ChatMessage {
  seq: number;
  channel: string;
  author: string;
  ts?: string;
  text: string;
  reply_to?: number;
  references?: string[];
}

/** Parse one chat message object into a RawRecord. */
export function loadChatMessage(m: ChatMessage, resolver: Resolver): RawRecord {
  const id = msgId(m.channel, m.seq);
  const refs: RawRef[] = [];

  // author -> SENT_BY
  if (m.author) refs.push(resolver.personRef("SENT_BY", m.author));

  // reply_to -> REPLY_TO (same channel)
  if (m.reply_to != null) {
    refs.push({
      rel: "REPLY_TO",
      targetId: msgId(m.channel, m.reply_to),
      name: `message ${m.reply_to}`,
    });
  }

  // explicit references[] + [[wikilink]] in text -> REFERENCES doc
  const seenDocRefs = new Set<string>();
  const addDocRef = (stem: string) => {
    const tid = docId(stem);
    if (seenDocRefs.has(tid)) return;
    seenDocRefs.add(tid);
    refs.push({ rel: "REFERENCES", targetId: tid, name: stem.replace(/^doc:/, "") });
  };
  for (const r of m.references ?? []) addDocRef(r);
  for (const wm of m.text.matchAll(WIKILINK_RE)) addDocRef(wm[1].trim());

  const props: Record<string, unknown> = { id, text: m.text, channel: m.channel };
  if (m.ts) props.ts = m.ts;

  return {
    kind: "chat",
    id,
    label: "Message",
    props,
    refs,
    text: m.text,
  };
}

/** Parse the raw chat.json contents into RawRecords. */
export function loadChatJson(raw: string, resolver: Resolver): RawRecord[] {
  const arr = JSON.parse(raw) as ChatMessage[];
  if (!Array.isArray(arr)) throw new Error("chat.json: expected a top-level array");
  return arr.map((m) => loadChatMessage(m, resolver));
}

/** Load chat.json from disk. */
export function loadChatFile(filePath: string, resolver: Resolver): RawRecord[] {
  return loadChatJson(readFileSync(filePath, "utf8"), resolver);
}
