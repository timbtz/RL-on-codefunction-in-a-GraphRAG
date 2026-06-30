// ============================================================
// loaders/meetings.ts — parse meetings.json (Teams standups) into RawRecords
// ============================================================
//
// Meeting shape:
//   { project, date, title, attendees[], discusses[], components[]?,
//     references[]?, transcript: [{speaker, text}, ...] }
//   id = meeting:<project>:<date>
//   attendees  -> ATTENDED person:<handle>
//   discusses  -> DISCUSSES ticket:<KEY>   (the project ticket talked through)
//   components -> ABOUT component:<slug>    (defaults to [project] when omitted)
//   references -> REFERENCES doc:<stem>
//   transcript -> body, split into per-utterance Chunks (1..n); the SOLUTION to a
//                 problem is named in this PROSE, not a structured field.

import { readFileSync } from "node:fs";
import {
  Resolver,
  docId,
  componentId,
  ticketId,
  meetingId,
  type RawRecord,
  type RawRef,
} from "../resolve";

export interface MeetingUtterance {
  speaker: string;
  text: string;
}

export interface MeetingRecord {
  project: string;
  date: string;
  title: string;
  attendees?: string[];
  discusses?: string[];
  components?: string[];
  references?: string[];
  transcript?: MeetingUtterance[];
}

/** Parse one meeting object into a RawRecord. */
export function loadMeeting(m: MeetingRecord, resolver: Resolver): RawRecord {
  const id = meetingId(m.project, m.date);
  const refs: RawRef[] = [];

  for (const a of m.attendees ?? []) refs.push(resolver.personRef("ATTENDED", a));
  for (const t of m.discusses ?? []) {
    refs.push({ rel: "DISCUSSES", targetId: ticketId(t), name: t.replace(/^ticket:/, "") });
  }
  const components = m.components && m.components.length ? m.components : [m.project];
  for (const c of components) {
    refs.push({ rel: "ABOUT", targetId: componentId(c), name: c });
  }
  for (const r of m.references ?? []) {
    refs.push({ rel: "REFERENCES", targetId: docId(r), name: r.replace(/^doc:/, "") });
  }

  // The transcript becomes the body (one paragraph per utterance) AND the
  // MENTIONS-scanned text (so @handles / PROJ-\d+ / component names get picked up).
  const utterances = (m.transcript ?? []).map(
    (u) => `${resolver.personName(u.speaker)}: ${u.text}`,
  );
  const body = utterances.join("\n\n");

  return {
    kind: "meeting",
    id,
    label: "Meeting",
    props: { id, title: m.title, date: m.date, project: m.project },
    refs,
    text: `${m.title}\n${body}`,
    body,
  };
}

/** Parse the raw meetings.json contents into RawRecords. */
export function loadMeetingsJson(raw: string, resolver: Resolver): RawRecord[] {
  const arr = JSON.parse(raw) as MeetingRecord[];
  if (!Array.isArray(arr)) throw new Error("meetings.json: expected a top-level array");
  return arr.map((m) => loadMeeting(m, resolver));
}

/** Load meetings.json from disk. */
export function loadMeetingsFile(filePath: string, resolver: Resolver): RawRecord[] {
  return loadMeetingsJson(readFileSync(filePath, "utf8"), resolver);
}
