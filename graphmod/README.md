# neo4j-graph-modules

Typed, modular graph schema system for Neo4j. Define graph subgraphs as **modules** with declarative schemas, hydrate them from a root node (eagerly or lazily), validate them against the schema inside a transaction, mutate them through domain services, and delete them recursively — all with full TypeScript inference.

## Core Concepts

### Module

A module owns a subgraph rooted at a single node type. `defineModule()` wires together:

- **Nodes** — each with `labels: [string, ...string[]]` and `properties` (typed)
- **Relationships** — typed edges between nodes with a `cardinality` (`"1"`, `"0..1"`, `"1..n"`, `"0..n"`) and optional edge `relProperties`. Every plain edge is also auto-exposed inbound on its target under `_reverse` (see [Reverse relationships](#reverse-relationships))
- **Cross-module refs** — edges that cross a module boundary. **Outbound** (a local node → a foreign module's node) and **inbound** (`inbound: true`; a foreign node → a local node). See [Cross-module relationships](#cross-module-relationships--where-to-declare-them)
- **`include`** — compose a foreign module's root node into this schema so plain relationships can reference it directly (Option B below)
- **onDelete** — custom teardown logic; the handler decides per node whether to delete it or merely unlink it (see [Deletion](#deletion))

A module exposes: `fromLazyGraph`, `delete`, `validate`, and `assertValid` (see [Module API](#module-api)).

### Property types

Node (and edge) properties map to TypeScript types. A trailing `?` marks optional; `[]` marks arrays:

`"string"`, `"number"`, `"boolean"`, `"string?"`, `"number?"`, `"boolean?"`, `"string[]"`, `"number[]"`

### Reverse relationships

Every plain relationship is **automatically** surfaced on its **target** node, inbound, under the node's `_reverse` map keyed by the relation name. There is nothing to declare:

```typescript
SPEAKS_IN: { from: "Person", to: "Chunk", cardinality: "0..n" }
```

The forward edge appears as `person.SPEAKS_IN` (a `Chunk[]`); the same edge appears inbound on the chunk as `chunk._reverse.SPEAKS_IN` (a `Person[]`). Both are typed and hydratable. The reverse direction is always `0..n` — a node may have any number of inbound edges of a given type, and the forward cardinality says nothing about the reverse, so reverse edges are never cardinality-constrained.

### Cross-module relationships — where to declare them

An edge can cross a module boundary. The question is *which module declares it*. The natural place is the module that owns the **content** the edge concerns — e.g. `Person ──SPEAKS_IN──> Chunk` belongs with `Chunk` (a content type), not with `Person` (a generic shared node that should not depend on every content module).

An **outbound** module ref only expresses an edge whose *source* is local:

```typescript
// In MeetingModule: a local Recording → the foreign Video module's root.
PROCESSED_AS: { from: "Recording", module: VideoModule, cardinality: "0..1" }
```

When the edge's *source* is the foreign node (`Person`) and its *target* is local (`Chunk`), there are two ways to still declare it in the content module:

**Option A — inbound module ref** (`inbound: true`). The foreign node stays in its own module; the edge is declared here and reached inbound under `_reverse`:

```typescript
// In VideoFileModule (owns Chunk); Person lives in PersonModule.
SPEAKS_IN: {
  inbound: true,
  module: PersonModule,
  from: "Person",      // node key *inside* PersonModule (the source)
  to: "Chunk",         // local node it lands on
  cardinality: "0..n",
  relProperties: { role: "string" },
}

await chunk._reverse.SPEAKS_IN            // Person[]  (PersonModule-hydrated)
await chunk._reverse.SPEAKS_IN.at(0)._rel.role
```

**Option B — `include` / compose.** Pull the foreign module's root node into this schema under an alias, then declare an ordinary plain relationship over it:

```typescript
// In VideoModule (owns Chunk).
include: { Person: PersonModule },
relationships: {
  SPEAKS_IN: { from: "Person", to: "Chunk", cardinality: "0..n" },
}

await chunk._reverse.SPEAKS_IN            // Person[]  (reverse)
await person.SPEAKS_IN                     // Chunk[]   (forward — Option B only)
```

#### A vs B

| | **A — inbound ref** | **B — `include`** |
|---|---|---|
| Edge lives in the content module | ✅ | ✅ |
| Foreign module stays standalone (no back-dependency) | ✅ | ✅ |
| `chunk._reverse.SPEAKS_IN` → foreign node | ✅ | ✅ |
| Forward `person.SPEAKS_IN` → local node | ❌ (inbound only) | ✅ (both directions) |
| Foreign node hydrates with | its **own** module's shape (properties + relations) | the foreign module's **properties** + *this* module's relations (a view) |
| Cardinality enforced | no — always unconstrained `0..n` inbound | yes — it's a normal plain rel, validated like any other |
| Cost | minimal; preserves a true module boundary | merges a foreign node into the local node set |

**Why both exist:** they trade module isolation against reachability. A keeps a clean boundary — the foreign node is only ever a fully-formed node from its own module, reached one way (inbound). B flattens the foreign node into this module so the edge behaves like any local relationship (both directions, cardinality-checked) at the cost of the included node being a *view* (foreign properties, local relations only).

**Which to pick:** reach for **A** when the foreign node is a generic shared entity you want to keep self-contained and you only need to walk *into* the local content (`chunk → its speakers`). Reach for **B** when you want the edge to behave like a first-class local relationship — traversable both ways and cardinality-validated — and you're fine with the foreign node hydrating as a local view. The two can coexist in one module, even per-relation.

### Hydration

`module.fromLazyGraph(session, rootId)` issues one query per node on demand. Every relationship and module ref is a memoized, chainable `LazyChain`-valued property that fetches on first access; inbound edges live under `_reverse`:

```typescript
const meeting = await MeetingModule.fromLazyGraph(session, id);
const speaker = await meeting.HAS_RECORDING.at(0).PROCESSED_AS.HAS_CHUNK.at(0)._reverse.SPEAKS_IN.at(0);
```

A per-call identity cache keyed by `${schemaName}::${id}` returns the same JS object whenever a node is re-encountered, so **cycles terminate** and identity comparisons hold across the tree. The cache is shared across cross-module hops. Chains are thenable proxies — `await` to resolve, or keep chaining; use `.at(i)` (or `[i]`) to index into arrays, and call array methods (`.map`, `.filter`) only *after* awaiting.

> Note: `onDelete` handlers receive the **lazy** hydrated root, so relationship access inside a handler must be `await`-ed.

### Validation

- `module.validate(data)` — checks cardinality against the schema on an already-hydrated (eager) object, returning a `ValidationError[]` (non-throwing).
- `module.assertValid(tx, rootId)` — runs the traversal query *inside the given transaction*, checks cardinality **and** required/typed properties, and **throws `SchemaValidationError`** on any mismatch. When called inside `session.executeWrite`, the throw aborts the transaction and rolls back every uncommitted write — including writes a parent service made before delegating. This is the primary guardrail for ingestion: write, then assert, atomically.

### Deletion

`module.delete(session, rootId)` lazily hydrates first (so the handler can inspect the graph), then runs the module's `onDelete`. Sub-modules are deleted via `ctx.deleteModule(SubModule, nodeId)`, which calls _that_ module's `onDelete` — recursion is driven by each module's own logic, not a generic traversal. Queued operations are flushed in a single `executeWrite` transaction.

There is no `ownership` field on the schema: whether a node is **owned** (deleted with the module via `ctx.deleteNode`) or **shared** (left in place, only its edges removed via `ctx.unlinkAll`) is decided entirely by the handler. Standalone entities like `Person` are simply never passed to `deleteNode` — the handler unlinks them and stops, while owned nodes such as `Chunk` or `Recording` are deleted.

## Setup

```bash
npm install
```

### Environment

Copy `.env.example` to `.env` and set your Neo4j connection (scripts load it via `dotenv/config`):

```
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="<password_here>"
NEO4J_DB="neo4j"
```

`neo4j-driver` is an **optional peer dependency** (`^5.28.3`) — the library only needs an object matching the [`GraphSession` interface](#graphsession-interface). Install the driver if you use the example wiring.

### Scripts

| Command | What it does |
|---|---|
| `npm run build` | Production build via tsup → `dist/` (CJS + ESM + `.d.ts`) |
| `npm run dev` | Runs `examples/usage.ts` with `tsx watch` + `dotenv/config` — re-executes on every save, no rebuild |
| `npm run example` | Single run of `examples/usage.ts` via tsx with `dotenv/config` |
| `npm run typecheck` | Type-check only (`tsc --noEmit`) |

### Build Output

tsup produces dual-format output (entry: `src/index.ts`, target `es2022`):

```
dist/
  index.js       # CommonJS
  index.mjs      # ES Modules
  index.d.ts     # CJS type declarations
  index.d.mts    # ESM type declarations
  *.map          # Source maps
```

The package `exports` field handles resolution automatically — `import` gets ESM, `require` gets CJS.

### Dev Workflow

During development, **skip the build entirely**. `tsx` compiles TypeScript on the fly:

```bash
# Watch mode — re-runs on any .ts file change
npm run dev

# Run any file directly
npx tsx -r dotenv/config examples/usage.ts
```

No `dist/` needed until you publish or consume the package from another project.

## File Structure

```
src/
  core/
    types.ts      Type primitives: PropType, NodeDef, RelationshipDef (+ reverse),
                  ModuleRef, ModuleSchema, Cardinality, HydratedNode, ResolveNode,
                  GraphSession, GraphTransaction, DeleteContext, DeleteHandler, AnyModule
    module.ts     defineModule() — factory wiring schema + fromGraph + fromLazyGraph
                  + delete + validate + assertValid
    cypher.ts     buildTraversalQuery() — generates the eager traversal Cypher
                  hydrateResult() — turns flat records into a nested typed tree
    lazy.ts       createLazyLoader() — per-node, cycle-safe lazy hydration; LazyChain proxy
    context.ts    createDeleteContext() — mutation queue for delete operations
                  createModuleInstance() — live handle with mutate/commit/refresh
    validate.ts   validateCardinality() — non-throwing cardinality check on hydrated data
                  assertValidInTx() — in-transaction cardinality + property validation
                  SchemaValidationError
  index.ts        Barrel export (public API + all modules)

  modules/
    video/
      video.module.ts    VideoModule (Document:Video → Chunk → Entity; Person SPEAKS_IN)
      video.service.ts   VideoService (ingestVideo + falsy-ingest test helpers)
    microsoft-teams/
      meeting/
        meeting.module.ts   MeetingModule (Meeting → Recording → [VideoModule], Chat → Msg → Attachment)
        meeting.service.ts  MeetingService (createDummyMeeting, addRecording, delete, deleteRecording)

examples/
  usage.ts                  End-to-end lifecycle scratchpad (wire your Neo4j session here)
  video-module-seed.cypher  Seed data for the video subgraph
```

## Defining a Module

```typescript
import { defineModule } from "neo4j-graph-modules";

const MyModule = defineModule({
  schema: {
    name: "MyModule",
    root: "RootNode",
    nodes: {
      RootNode:  { labels: ["Root"],  properties: { id: "string", title: "string" } },
      Child:     { labels: ["Child"], properties: { value: "number" } },
      SharedRef: { labels: ["Ref"],   properties: { id: "string", name: "string?" } },
    },
    relationships: {
      HAS_CHILD: { from: "RootNode", to: "Child", cardinality: "1..n" },
      // Plain edge; auto-exposed inbound on SharedRef as `sharedRef._reverse.REFS`.
      REFS: { from: "Child", to: "SharedRef", cardinality: "0..n" },
      // Cross-module refs live in `relationships` too. Outbound (local source):
      LINKS_TO: { from: "Child", module: OtherModule, cardinality: "0..1" },
      // Inbound (foreign source → local node) would add `inbound: true` + `from`/`to`.
    },
  },

  // onDelete receives the LAZY hydrated root — await relationship access.
  async onDelete(ctx, data) {
    for (const child of await data.HAS_CHILD) {
      const link = await child.LINKS_TO;
      if (link) await ctx.deleteModule(OtherModule, link._id);
      ctx.unlinkAll(child._id, "REFS", "out");
      ctx.deleteNode(child._id);
    }
    ctx.deleteNode(data._id);
  },
});
```

## Module API

| Method | Returns | Notes |
|---|---|---|
| `fromLazyGraph(session, rootId)` | `LazyChain<LazyHydrated>` | Lazy, chainable, cycle-safe; one query per node |
| `delete(session, rootId)` | `Promise<void>` | Lazily hydrates, runs `onDelete`, flushes in one tx |
| `validate(data)` | `ValidationError[]` | Non-throwing cardinality check on eager data |
| `assertValid(tx, rootId)` | `Promise<void>` | In-tx cardinality + property check; throws `SchemaValidationError` (rolls back the tx) |

## Domain Services

Wrap a module with a service class for business operations. The bundled `VideoService` / `MeetingService` show the pattern: each write method accepts a `GraphSession` **or** a `GraphTransaction`, so it can run standalone *or* enlist in a caller's transaction, and validates with `assertValid` before committing.

```typescript
class VideoService {
  static async ingestVideo(
    target: GraphSession | GraphTransaction,
    input: IngestVideoInput,
    options: IngestVideoOptions = {},
  ): Promise<string> {
    const docId = crypto.randomUUID();
    const work = async (tx: GraphTransaction) => {
      await VideoService.writeGraph(tx, docId, input, options);
      await VideoModule.assertValid(tx, docId); // throws + rolls back on schema violation
    };
    if ("executeWrite" in target) await target.executeWrite(work);
    else await work(target);
    return docId;
  }
}
```

Because a service can take a transaction, services compose: `MeetingService.addRecording` creates a `Recording`, calls `VideoService.ingestVideo` on the **same** transaction (linking the new video via `PROCESSED_AS`), then asserts the whole `Meeting` subgraph — all atomically.

For mutate/commit/refresh-style live handles, `createModuleInstance(module, session, rootId, data)` returns a `ModuleInstance` with a queued `mutate()`, a `commit()` (single `executeWrite`), and a `refresh()` that re-hydrates `data`.

## Type Inference

The hydrated type is inferred entirely from the schema — no manual type declarations. Each edge is wrapped in `LazyChain<…>`; `await` resolves it:

```typescript
const data = await MyModule.fromLazyGraph(session, "id-1");
data.title;                                // string         (RootNode.properties)
const child = await data.HAS_CHILD.at(0);  // Child          (cardinality "1..n")
const refs  = await child.REFS;            // SharedRef[]    (cardinality "0..n")
const link  = await child.LINKS_TO;        // OtherModule hydrated | null  ("0..1")
const who   = await refs[0]._reverse.REFS; // Child[]        (auto-reverse, inbound)
```

Helpers `InferModuleHydrated<M>` / `InferModuleLazyHydrated<M>` recover a module's hydrated type for reuse.

## DeleteContext API

Available inside `onDelete` handlers (`ctx`):

| Method | Effect |
|---|---|
| `ctx.unlinkAll(nodeId, relType, "in"\|"out"\|"both")` | Remove all edges of a type in a direction |
| `ctx.deleteNode(nodeId)` | `DETACH DELETE` an owned node |
| `ctx.deleteModule(module, rootId)` | Flush pending ops, then recursively run another module's `onDelete` |
| `ctx.mutate(cypher, params)` | Queue arbitrary Cypher |
| `ctx.flush()` | Execute all queued operations in a single `executeWrite` transaction |

## GraphSession Interface

Any object matching this interface works — you don't need the full `neo4j-driver`. Note that writes go through `executeWrite`, so the session must support it (a bare transaction supports `run` only):

```typescript
interface GraphTransaction {
  run(cypher: string, params?: Record<string, unknown>): Promise<{
    records: Array<{ get(key: string): any; keys: string[]; toObject(): Record<string, any> }>;
  }>;
}

interface GraphSession extends GraphTransaction {
  executeWrite<T>(work: (tx: GraphTransaction) => Promise<T>): Promise<T>;
}
```

## Bundled modules

- **VideoModule** (`Video`, root `Document:Video`) — `Document ─HAS_CHUNK→ Chunk ─MENTIONS→ Entity`, with `Person ─SPEAKS_IN→ Chunk`. Demonstrates **Option B** (`include: { Person: PersonModule }`), so `SPEAKS_IN` is a plain rel — forward as `person.SPEAKS_IN`, inbound as `chunk._reverse.SPEAKS_IN`. `Person`/`Entity` are treated as shared by the delete handler; deletion unlinks them and removes the `Document`/`Chunk` nodes plus inbound `PROCESSED_AS`.
- **VideoFileModule** (`VideoFileModule`, root `Document:VideoFile`) — `Document ─HAS_CHUNK→ Chunk`, with the same `Person ─SPEAKS_IN→ Chunk` edge declared via **Option A** (inbound module ref into `PersonModule`), reached as `chunk._reverse.SPEAKS_IN`.
- **PersonModule** (`PersonModule`, root `Person`) — a thin, standalone shared node with no content relations; composed/referenced by content modules via Option A or B.
- **MeetingModule** (`MSTeams_Meeting`, root `Meeting`) — organizers/invitees (`Person`), `Recording`s (each optionally `PROCESSED_AS` a `VideoModule` subgraph), and a `Chat` of `Msg`s with `Attachment`s and `REPLIED_TO` threading. Deletion recurses into video subgraphs and tears down the meeting's own nodes while leaving `Person` nodes intact.
