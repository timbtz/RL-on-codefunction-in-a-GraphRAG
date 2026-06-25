// ============================================================
// types.ts — Core primitives for the graph module system
// ============================================================

// ---- Property types (schema → runtime mapping) ----

export type PropType =
  | "string"
  | "number"
  | "boolean"
  | "datetime"
  | "string?"
  | "number?"
  | "boolean?"
  | "datetime?"
  | "string[]"
  | "number[]";

/**
 * Structural shape of a Neo4j temporal (`DateTime`/`Date`) value as returned by
 * the driver. Duck-typed so we don't take a hard dependency on `neo4j-driver`,
 * matching the rest of this module. `.toString()` yields an ISO 8601 string.
 */
export interface Neo4jDateTime {
  year: number | { low: number; high: number };
  month: number | { low: number; high: number };
  day: number | { low: number; high: number };
  toString(): string;
}

export type ResolveProp<T extends PropType> = T extends "string"
  ? string
  : T extends "number"
    ? number
    : T extends "boolean"
      ? boolean
      : T extends "datetime"
        ? Neo4jDateTime
        : T extends "string?"
          ? string | undefined
          : T extends "number?"
            ? number | undefined
            : T extends "boolean?"
              ? boolean | undefined
              : T extends "datetime?"
                ? Neo4jDateTime | undefined
                : T extends "string[]"
                  ? string[]
                  : T extends "number[]"
                    ? number[]
                    : never;

export type ResolveProps<P extends Record<string, PropType>> = {
  [K in keyof P]: ResolveProp<P[K]>;
};

// ---- Edge (relationship) properties ----
//
// Turns a relationship def's declared `properties` into a `_rel` block that is
// intersected onto the *child* node at each traversal hop. A rel with no
// declared properties contributes `{}` (no `_rel` key), keeping access
// non-breaking.

export type EdgeProps<R> = R extends {
  relProperties: infer P extends Record<string, PropType>;
}
  ? { _rel: ResolveProps<P> }
  : {};

// ---- Cardinality ----

export type Cardinality = "1" | "0..1" | "1..n" | "0..n";

export type CardinalityWrap<T, C extends Cardinality> = C extends "1"
  ? T
  : C extends "0..1"
    ? T | null
    : C extends "1..n"
      ? [T, ...T[]]
      : C extends "0..n"
        ? T[]
        : never;

// ---- Node definition ----

export interface NodeDef {
  labels: [string, ...string[]];
  properties: Record<string, PropType>;
}

// ---- Relationship definition ----

export interface RelationshipDef<
  TNodes extends Record<string, NodeDef> = Record<string, NodeDef>,
> {
  from: keyof TNodes & string;
  to: keyof TNodes & string;
  cardinality: Cardinality;
  /** Properties carried on the relationship edge itself (exposed as `_rel`). */
  relProperties?: Record<string, PropType>;
}

// ---- Module reference (cross-module edge) ----

export interface ModuleRef<
  TNodes extends Record<string, NodeDef> = Record<string, NodeDef>,
> {
  from: keyof TNodes & string;
  module: AnyModule;
  /**
   * Node key *inside the target module* this edge lands on. Defaults to the
   * target module's `root` when omitted. Lets a ref point at a sub-node
   * (e.g. `Person -> Chunk`) rather than only the foreign root.
   */
  to?: string;
  cardinality: Cardinality;
  /** Properties carried on the cross-module edge itself (exposed as `_rel`). */
  relProperties?: Record<string, PropType>;
}

// ---- Inbound module reference (foreign source → local node) ----
//
// An outbound `ModuleRef` can only express an edge whose *source* is a local
// node. An `InboundModuleRef` expresses the opposite: an edge whose **source**
// lives in a foreign module and which lands on a *local* node. This lets the
// edge be declared in the module that owns the content it concerns (e.g.
// `SPEAKS_IN` declared in the VideoFile module even though `Person` is foreign)
// without that foreign module having to depend on every content module.
//
// The edge is reached inbound, under the local node's `_reverse` map, keyed by
// the relation name — `chunk._reverse.SPEAKS_IN` yields the foreign-hydrated
// source nodes. Like auto-reverse edges it is unconstrained (`0..n`).
export interface InboundModuleRef<
  TNodes extends Record<string, NodeDef> = Record<string, NodeDef>,
> {
  /** Discriminator: marks this entry as an inbound cross-module edge. */
  inbound: true;
  /** The foreign module that owns the source node. */
  module: AnyModule;
  /** Node key *inside the foreign module* the edge originates from. */
  from: string;
  /** Local node key the edge lands on. */
  to: keyof TNodes & string;
  cardinality: Cardinality;
  /** Properties carried on the cross-module edge itself (exposed as `_rel`). */
  relProperties?: Record<string, PropType>;
}

// ---- Merged relationship entry ----
//
// A single `relationships` map holds every kind of edge. An entry is a plain
// relationship (`RelationshipDef`) unless it carries a `module`, in which case
// it is a cross-module reference: an `InboundModuleRef` when it also carries
// `inbound: true` (foreign source → local node), otherwise an outbound
// `ModuleRef` (local source → foreign node). The presence/shape of `module` is
// the only structural difference, and it is what every type- and runtime-level
// branch keys off.

export type RelOrModule<
  TNodes extends Record<string, NodeDef> = Record<string, NodeDef>,
> = RelationshipDef<TNodes> | ModuleRef<TNodes> | InboundModuleRef<TNodes>;

/**
 * True when a merged relationship entry crosses a module boundary (either
 * direction). Used to exclude such entries from plain-relationship handling.
 */
export function isModuleRef(
  def: RelOrModule,
): def is ModuleRef | InboundModuleRef {
  return "module" in def && def.module != null;
}

/** True when a cross-module entry is *inbound* (foreign source → local node). */
export function isInboundModuleRef(def: RelOrModule): def is InboundModuleRef {
  return (
    "module" in def &&
    def.module != null &&
    (def as { inbound?: unknown }).inbound === true
  );
}

// ---- Schema ----

export interface ModuleSchema<
  TNodes extends Record<string, NodeDef> = Record<string, NodeDef>,
  TRels extends Record<string, RelOrModule<TNodes>> = Record<
    string,
    RelOrModule<TNodes>
  >,
> {
  name: string;
  root: keyof TNodes & string;
  nodes: TNodes;
  /**
   * Compose foreign modules' root nodes into this schema's node set. Each entry
   * `alias: ForeignModule` merges that module's *root* node def in under
   * `alias`, after which plain relationships may reference `alias` as a normal
   * local node (in either direction). The included node hydrates with the
   * foreign module's properties but *this* module's relationships — a view, not
   * a full standalone sub-graph. Precise per-alias typing is applied by
   * `defineModule`; the interface keeps it loose.
   */
  include?: Record<string, AnyModule>;
  relationships: TRels;
}

// ---- Resolved node (base shape of any hydrated node) ----

export type ResolveNode<N extends NodeDef> = {
  _id: string;
  _labels: string[];
} & ResolveProps<N["properties"]>;

// ---- Hydrated node: resolved props + nested relations + sub-modules ----

export type HydratedNode<
  TNodes extends Record<string, NodeDef>,
  TRels extends Record<string, RelOrModule<TNodes>>,
  TKey extends keyof TNodes & string,
> = ResolveNode<TNodes[TKey]> &
  // Forward relationships (plain rels: from === TKey, no `module`)
  {
    [R in keyof TRels as TRels[R] extends { module: any }
      ? never
      : TRels[R] extends { from: TKey }
        ? R
        : never]: TRels[R] extends {
      to: infer To extends keyof TNodes & string;
      cardinality: infer C extends Cardinality;
    }
      ? CardinalityWrap<
          HydratedNode<TNodes, TRels, To> & EdgeProps<TRels[R]>,
          C
        >
      : never;
  } & {
    // Auto-reverse + inbound module refs: every edge whose `to` is this node is
    // exposed inbound under `_reverse`, keyed by the relation name. Always a
    // collection (`0..n`) — a node may have any number of inbound edges of a
    // given type, and the forward cardinality says nothing about the reverse.
    _reverse: {
      // Plain rels whose `to` is this node.
      [R in keyof TRels as TRels[R] extends { module: any }
        ? never
        : TRels[R] extends { to: TKey }
          ? R
          : never]: TRels[R] extends {
        from: infer From extends keyof TNodes & string;
      }
        ? CardinalityWrap<
            HydratedNode<TNodes, TRels, From> & EdgeProps<TRels[R]>,
            "0..n"
          >
        : never;
    } & {
      // Inbound module refs whose local `to` is this node — the source is a
      // foreign node, hydrated by its own module.
      [R in keyof TRels as TRels[R] extends { inbound: true; to: TKey }
        ? R
        : never]: TRels[R] extends { module: infer Mod }
        ? CardinalityWrap<InferModuleHydrated<Mod> & EdgeProps<TRels[R]>, "0..n">
        : never;
    };
  } & {
    // Outbound cross-module relationships (carry a `module`, not inbound)
    [M in keyof TRels as TRels[M] extends { inbound: true }
      ? never
      : TRels[M] extends { from: TKey; module: any }
        ? M
        : never]: TRels[M] extends {
      module: infer Mod;
      cardinality: infer C extends Cardinality;
    }
      ? CardinalityWrap<InferModuleHydrated<Mod> & EdgeProps<TRels[M]>, C>
      : never;
  };

// ---- Infer the eager hydrated shape of any module ----
// Anchored on `validate`'s parameter (the module's `HydratedNode`) since the
// eager `fromGraph` entry point no longer exists; `fromLazyGraph` is canonical.

export type InferModuleHydrated<M> =
  M extends { validate: (data: infer H) => any } ? H : never;

// ---- Minimal session interface (compatible with neo4j-driver) ----

export interface GraphTransaction {
  run(
    cypher: string,
    params?: Record<string, unknown>,
  ): Promise<{
    records: Array<{
      get(key: string): any;
      keys: string[];
      toObject(): Record<string, any>;
    }>;
  }>;
}

export interface GraphSession extends GraphTransaction {
  executeWrite<T>(work: (tx: GraphTransaction) => Promise<T>): Promise<T>;
}

// ---- Forward reference for AnyModule (used in ModuleRef) ----

export interface AnyModule {
  schema: ModuleSchema<any, any>;
  fromLazyGraph(session: GraphSession, rootId: string): PromiseLike<any>;
  delete(session: GraphSession, rootId: string): Promise<void>;
  validate(data: any): unknown;
  onDelete: DeleteHandler<any>;
}

// ---- Delete handler ----

export type DeleteHandler<THydrated> = (
  ctx: DeleteContext,
  data: THydrated,
) => Promise<void>;

export interface DeleteContext {
  session: GraphSession;
  /** Remove all relationships (incoming and outgoing) from/to a node by the id property */
  unlinkAll(nodeId: string): void;
  /** Mark an owned node for deletion (runs on flush) */
  deleteNode(nodeId: string): void;
  /** Recursively delete a sub-module's subgraph using its own onDelete */
  deleteModule(module: AnyModule, nodeId: string): Promise<void>;
  /** Queue raw Cypher for edge cases */
  mutate(cypher: string, params: Record<string, unknown>): void;
  /** Flush all queued operations to the database */
  flush(): Promise<void>;
}
