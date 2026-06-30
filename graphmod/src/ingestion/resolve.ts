// ============================================================
// resolve.ts — canonical-id helpers, controlled-vocabulary gazetteer,
//              and the shared RawRecord intermediate representation.
// ============================================================
//
// Every node in the corpus graph is keyed by a *canonical id* so that
// re-ingesting the same corpus MERGEs onto the same node (idempotent writes,
// no duplicates). The id scheme mirrors the CONTRACT:
//
//   Person    person:<slug(handle|name)>
//   Component component:<slug>
//   Entity    entity:<slug>
//   Chunk     <ownerId>#c<index>
//   Document  doc:<filename-stem>
//   Message   msg:<channel>:<seq>
//   Ticket    ticket:<KEY>           (KEY kept verbatim, e.g. ticket:PROJ-12)
//
// The gazetteer is the small controlled vocabulary of known people/components
// (built from names + aliases). It resolves @handles and bare name mentions in
// free text to their canonical Person/Component id + display name, which the
// extractor uses for MENTIONS edges and for naming entities inside chunk cards.

// ---- Relationship type names (verbatim from the CONTRACT) ----

export type RelType =
  | "AUTHORED_BY"
  | "ABOUT"
  | "REFERENCES"
  | "SENT_BY"
  | "REPLY_TO"
  | "ASSIGNED"
  | "REPORTED"
  | "BLOCKS"
  // --- Teams meeting (Meeting) ---
  | "ATTENDED"      // Meeting -> Person
  | "DISCUSSES"     // Meeting -> Ticket (the project ticket talked through in the standup)
  // --- GitHub (PullRequest) ---
  | "REVIEWED_BY"   // PullRequest -> Person
  | "FIXES";        // PullRequest -> Ticket (the PR that resolves the ticket)

/** Owner (root) label for each source kind. */
export type SourceKind = "doc" | "chat" | "jira" | "meeting" | "github";
export type OwnerLabel = "Document" | "Message" | "Ticket" | "Meeting" | "PullRequest";

/** The Neo4j label a structured edge points AT, keyed by relationship type. */
export const REL_TARGET_LABEL: Record<RelType, string> = {
  AUTHORED_BY: "Person",
  SENT_BY: "Person",
  ASSIGNED: "Person",
  REPORTED: "Person",
  ATTENDED: "Person",
  REVIEWED_BY: "Person",
  ABOUT: "Component",
  REFERENCES: "Document",
  BLOCKS: "Ticket",
  DISCUSSES: "Ticket",
  FIXES: "Ticket",
  REPLY_TO: "Message",
};

// ---- Intermediate representation produced by the loaders ----

/**
 * A single structured edge discovered by a loader, already canonicalized.
 * `targetId` is the canonical node id of the edge target; `name` is the human
 * label used when this edge is serialized into the owner's chunk "card"
 * (e.g. a Person's display name, a Component slug, a Document stem, a Ticket
 * key). Keeping resolution in the loader keeps the extractor purely mechanical.
 */
export interface RawRef {
  rel: RelType;
  targetId: string;
  /** display name for the chunk card (omit from cards when empty) */
  name: string;
}

/**
 * Source-agnostic record emitted by every loader and consumed by the extractor.
 * One RawRecord == one Document / Message / Ticket root node.
 */
export interface RawRecord {
  kind: SourceKind;
  /** canonical owner id: docId() / msgId() / ticketId() */
  id: string;
  label: OwnerLabel;
  /** node properties to MERGE (already shaped per the schema module) */
  props: Record<string, unknown>;
  /** structured references (resolved to canonical ids by the loader) */
  refs: RawRef[];
  /** prose scanned for regex/gazetteer MENTIONS (and the card body for docs) */
  text: string;
  /** doc body split into chunks 1..n; undefined for chat/jira (single card) */
  body?: string;
}

// ---- slug + canonical ids ----

/** lowercase, non-alnum → "-", collapse repeats, trim. */
export function slug(x: string): string {
  return x
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

export function personId(handleOrName: string): string {
  return `person:${slug(handleOrName)}`;
}

export function componentId(s: string): string {
  return `component:${slug(s)}`;
}

export function entityId(name: string): string {
  return `entity:${slug(name)}`;
}

/** Accepts either a bare stem ("auth-redesign") or a full id ("doc:auth-redesign"). */
export function docId(stem: string): string {
  const s = stem.trim();
  return s.startsWith("doc:") ? `doc:${slug(s.slice(4))}` : `doc:${slug(s)}`;
}

export function msgId(channel: string, seq: number | string): string {
  return `msg:${channel}:${seq}`;
}

/** Ticket KEY is kept verbatim (uppercase), e.g. ticketId("PROJ-12") → "ticket:PROJ-12". */
export function ticketId(key: string): string {
  const k = key.trim();
  return k.startsWith("ticket:") ? k : `ticket:${k}`;
}

/** Teams meeting id, e.g. meetingId("billing","2026-05-06") → "meeting:billing:2026-05-06". */
export function meetingId(project: string, date: string): string {
  const p = project.trim();
  if (p.startsWith("meeting:")) return p;
  return `meeting:${slug(project)}:${date.trim()}`;
}

/** GitHub PR id, e.g. prId("billing", 88) → "pr:billing#88". */
export function prId(repo: string, number: number | string): string {
  const r = repo.trim();
  if (r.startsWith("pr:")) return r;
  return `pr:${slug(repo)}#${number}`;
}

/** Chunk id from its owner id + 0-based index. */
export function chunkId(ownerId: string, index: number): string {
  return `${ownerId}#c${index}`;
}

// ---- Controlled vocabulary ----

export interface PersonVocab {
  handle: string;
  name: string;
}
export interface ComponentVocab {
  slug: string;
  name: string;
}

/** Known people in the corpus. */
export const KNOWN_PEOPLE: PersonVocab[] = [
  { handle: "tim", name: "Tim" },
  { handle: "lasse", name: "Lasse" },
  { handle: "mara", name: "Mara" },
  { handle: "noah", name: "Noah" },
  { handle: "priya", name: "Priya" },
  { handle: "omar", name: "Omar" },
  { handle: "sara", name: "Sara" },
  { handle: "jack", name: "Jack" },
];

/** Known components / projects. */
export const KNOWN_COMPONENTS: ComponentVocab[] = [
  { slug: "auth-redesign", name: "auth-redesign" },
  { slug: "billing", name: "billing" },
  { slug: "search-index", name: "search-index" },
  { slug: "onboarding", name: "onboarding" },
  { slug: "infra", name: "infra" },
  { slug: "data-platform", name: "data-platform" },
];

// ---- Gazetteer + resolver ----

export interface GazetteerHit {
  id: string;
  label: "Person" | "Component";
  name: string;
}

interface GazetteerTerm {
  /** lowercased surface form to match in free text */
  term: string;
  hit: GazetteerHit;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * The entity resolver: holds the controlled-vocabulary gazetteer plus name
 * lookups. Built once per ingest and shared across loaders + extractor.
 */
export class Resolver {
  private readonly terms: GazetteerTerm[] = [];
  private readonly personNameByHandle = new Map<string, string>();

  constructor(
    people: PersonVocab[] = KNOWN_PEOPLE,
    components: ComponentVocab[] = KNOWN_COMPONENTS,
  ) {
    for (const p of people) {
      this.personNameByHandle.set(slug(p.handle), p.name);
      const hit: GazetteerHit = {
        id: personId(p.handle),
        label: "Person",
        name: p.name,
      };
      for (const t of [p.handle, `@${p.handle}`, p.name]) {
        this.terms.push({ term: t.toLowerCase(), hit });
      }
    }
    for (const c of components) {
      const hit: GazetteerHit = {
        id: componentId(c.slug),
        label: "Component",
        name: c.name,
      };
      for (const t of [c.slug, c.name]) {
        this.terms.push({ term: t.toLowerCase(), hit });
      }
    }
    // Longest term first so multi-word/hyphenated forms win over substrings.
    this.terms.sort((a, b) => b.term.length - a.term.length);
  }

  /** Display name for a person handle (controlled vocab, else capitalized). */
  personName(handle: string): string {
    const key = slug(handle);
    const known = this.personNameByHandle.get(key);
    if (known) return known;
    return key
      .split("-")
      .filter(Boolean)
      .map((w) => w[0].toUpperCase() + w.slice(1))
      .join(" ");
  }

  /** Build a resolved Person RawRef (always resolves, creating an id for unknown handles). */
  personRef(rel: RelType, handle: string): RawRef {
    return { rel, targetId: personId(handle), name: this.personName(handle) };
  }

  /** Resolve an @handle / bare token to a canonical Person id + display name. */
  resolvePersonHandle(handle: string): GazetteerHit {
    return {
      id: personId(handle),
      label: "Person",
      name: this.personName(handle),
    };
  }

  /**
   * Scan free text for controlled-vocabulary mentions (person names/handles +
   * component slugs). Returns one hit per distinct canonical id.
   */
  gazetteerMentions(text: string): GazetteerHit[] {
    const lower = text.toLowerCase();
    const seen = new Set<string>();
    const out: GazetteerHit[] = [];
    for (const { term, hit } of this.terms) {
      if (seen.has(hit.id)) continue;
      // Word-ish boundary that tolerates a leading "@" but treats a hyphen as
      // PART of a token, not a boundary -- otherwise a slug that prefixes a longer
      // hyphenated token false-matches (e.g. "search-index" inside
      // "search-index-design"), attaching a spurious component MENTIONS. Excluding
      // "-" from the boundary class keeps internal hyphens inside the term while
      // still matching the slug when it stands alone (followed by space/./EOL).
      const re = new RegExp(`(?:^|[^a-z0-9-])${escapeRegExp(term)}(?:$|[^a-z0-9-])`);
      if (re.test(lower)) {
        seen.add(hit.id);
        out.push(hit);
      }
    }
    return out;
  }

  /** The canonical Person/Component ids the LLM lever must dedupe against. */
  canonicalEntityIds(): Set<string> {
    const ids = new Set<string>();
    for (const { hit } of this.terms) ids.add(hit.id);
    return ids;
  }
}
