// ============================================================
// lazy.ts — fromLazyGraph: per-node, cycle-safe hydration
// ============================================================
//
// One Cypher query per node fetches scalar properties plus the
// neighbour IDs for every forward rel, reverse rel, and module ref.
// Each edge is installed as a memoized Promise-valued getter on the
// node, so `await node.SPEAKERS` triggers the next query lazily.
//
// A per-call cache keyed by `${schemaName}::${id}` returns the same
// JS object whenever a node is re-encountered, so cycles terminate
// (the second visit awaits the same in-flight Promise) and identity
// comparisons work across the tree.
// ============================================================

import type {
  Cardinality,
  GraphSession,
  ModuleSchema,
  NodeDef,
  PropType,
  RelOrModule,
  ResolveNode,
  CardinalityWrap,
  EdgeProps,
  AnyModule,
} from "./types.js";
import { isModuleRef, isInboundModuleRef } from "./types.js";

// ---- Public types ----

/**
 * A thenable Proxy that forwards property access into a Promise chain.
 *
 * `await chain.A.B.C` resolves the underlying promise(s) end-to-end.
 * Use `.at(i)` (or numeric index `[0]`) to dive into an array element.
 *
 * Any non-trivial method calls (`.map`, `.filter`, etc.) should be made
 * after `await`-ing — only `.at(i)` is special-cased on chains.
 */
export type LazyChain<T> = PromiseLike<T> & ChainAccessors<Awaited<T>>;

type ChainAccessors<T> = NonNullable<T> extends readonly (infer U)[]
  ? {
      at(index: number): LazyChain<U>;
      readonly [index: number]: LazyChain<U>;
    }
  : NonNullable<T> extends object
    ? { readonly [K in keyof NonNullable<T>]: LazyChain<Awaited<NonNullable<T>[K]>> }
    : {};

export type LazyHydratedNode<
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
      ? LazyChain<
          CardinalityWrap<
            LazyHydratedNode<TNodes, TRels, To> & EdgeProps<TRels[R]>,
            C
          >
        >
      : never;
  } & {
    // Auto-reverse + inbound module refs: every edge whose `to` is this node,
    // exposed inbound under `_reverse` keyed by relation name (always `0..n`).
    _reverse: {
      // Plain rels whose `to` is this node.
      [R in keyof TRels as TRels[R] extends { module: any }
        ? never
        : TRels[R] extends { to: TKey }
          ? R
          : never]: TRels[R] extends {
        from: infer From extends keyof TNodes & string;
      }
        ? LazyChain<
            CardinalityWrap<
              LazyHydratedNode<TNodes, TRels, From> & EdgeProps<TRels[R]>,
              "0..n"
            >
          >
        : never;
    } & {
      // Inbound module refs whose local `to` is this node — the foreign source
      // is hydrated by its own module (resolved via `__lazyNodes`).
      [R in keyof TRels as TRels[R] extends { inbound: true; to: TKey }
        ? R
        : never]: TRels[R] extends { module: infer Mod }
        ? LazyChain<
            CardinalityWrap<
              ResolveInboundModuleLazyNode<Mod, TRels[R]> & EdgeProps<TRels[R]>,
              "0..n"
            >
          >
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
      ? LazyChain<
          CardinalityWrap<
            ResolveModuleLazyNode<Mod, TRels[M]> & EdgeProps<TRels[M]>,
            C
          >
        >
      : never;
  };

export type InferModuleLazyHydrated<M> = M extends {
  fromLazyGraph: (...args: any[]) => PromiseLike<infer H>;
}
  ? H
  : never;

/**
 * Resolve the lazily-hydrated shape a cross-module ref lands on. When the ref
 * names a `to` node, look it up in the target module's phantom `__lazyNodes`
 * map; otherwise fall back to the module's root (its `fromLazyGraph` return).
 */
export type ResolveModuleLazyNode<Mod, Ref> = Ref extends {
  to: infer K extends string;
}
  ? Mod extends { __lazyNodes?: infer NM }
    ? K extends keyof NM
      ? NM[K]
      : never
    : never
  : InferModuleLazyHydrated<Mod>;

/**
 * Resolve the lazily-hydrated shape an *inbound* module ref originates from.
 * The foreign source node is named by the ref's `from`; look it up in the
 * target module's phantom `__lazyNodes` map.
 */
export type ResolveInboundModuleLazyNode<Mod, Ref> = Ref extends {
  from: infer K extends string;
}
  ? Mod extends { __lazyNodes?: infer NM }
    ? K extends keyof NM
      ? NM[K]
      : never
    : never
  : never;

// ---- Chain factory ----

function chain<T>(promise: Promise<T>): LazyChain<T> {
  const handler: ProxyHandler<object> = {
    get(_target, prop) {
      if (prop === "then") {
        return (onFul: any, onRej: any) => promise.then(onFul, onRej);
      }
      if (prop === "catch") {
        return (onRej: any) => promise.catch(onRej);
      }
      if (prop === "finally") {
        return (onFin: any) => promise.finally(onFin);
      }
      // Symbols (Symbol.toPrimitive, util.inspect, etc.) get a clean undefined
      // so inspectors don't trigger an infinite chain of property accesses.
      if (typeof prop === "symbol") return undefined;
      if (prop === "at") {
        return (index: number) =>
          chain(
            promise.then((v: any) =>
              Array.isArray(v) ? v[index < 0 ? v.length + index : index] : v?.[index],
            ),
          );
      }
      if (/^-?\d+$/.test(prop)) {
        const idx = Number(prop);
        return chain(
          promise.then((v: any) =>
            Array.isArray(v) ? v[idx < 0 ? v.length + idx : idx] : v?.[idx],
          ),
        );
      }
      // Default: forward into the promise chain. If `v[prop]` is itself a
      // LazyChain (because edges return chains), Promise.then auto-unwraps
      // it, so chains flatten across nested edge hops.
      return chain(promise.then((v: any) => v?.[prop]));
    },
  };
  return new Proxy({}, handler) as unknown as LazyChain<T>;
}

// ---- Internals ----

type LazyCache = Map<string, Promise<any>>;

interface NodeQueryPlan {
  cypher: string;
  /** Outgoing edges to install on the node, in `RETURN` order. */
  edges: EdgePlan[];
}

interface EdgePlan {
  /** Property name to install on the hydrated node. */
  propName: string;
  /**
   * Alias of the result column. Holds `collect(DISTINCT t.id)` (a `string[]`)
   * for plain edges, or `collect(DISTINCT {id, rel})` (a `{id, rel}[]`) when
   * `relProps` is set.
   */
  idsAlias: string;
  cardinality: Cardinality;
  /**
   * When set, the edge declares its own properties: the column carries
   * `{id, rel}` pairs and each loaded child is wrapped with a `_rel` block.
   */
  relProps?: Record<string, PropType>;
  /** Inbound (auto-reverse) edges install under the node's `_reverse` map. */
  reverse?: boolean;
  /** Loader for each target ID. */
  load: (
    session: GraphSession,
    cache: LazyCache,
    targetId: string,
  ) => Promise<any>;
}

const planCache = new WeakMap<ModuleSchema<any, any>, Map<string, NodeQueryPlan>>();

function getNodePlan(
  schema: ModuleSchema<any, any>,
  nodeKey: string,
  loaderFactory: (
    nodeKey: string,
  ) => (session: GraphSession, cache: LazyCache, id: string) => Promise<any>,
): NodeQueryPlan {
  let perSchema = planCache.get(schema);
  if (!perSchema) {
    perSchema = new Map();
    planCache.set(schema, perSchema);
  }
  const cached = perSchema.get(nodeKey);
  if (cached) return cached;

  const nodeDef = schema.nodes[nodeKey] as NodeDef;
  if (!nodeDef) throw new Error(`Unknown node key "${nodeKey}" in module "${schema.name}"`);
  const label = nodeDef.labels[0];

  const edges: EdgePlan[] = [];
  const withParts: string[] = ["n"];
  const matches: string[] = [];
  let i = 0;

  // Forward relationships from this node (plain rels only; module refs below)
  for (const [relName, rel] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    if (isModuleRef(rel) || rel.from !== nodeKey) continue;
    const toDef = schema.nodes[rel.to] as NodeDef;
    const idsAlias = `ids${i++}`;
    const relProps = rel.relProperties;
    const collected = relProps
      ? `collect(DISTINCT {id: t.id, rel: properties(r${idsAlias})})`
      : `collect(DISTINCT t.id)`;
    const relVar = relProps ? `r${idsAlias}` : "";
    matches.push(
      `OPTIONAL MATCH (n)-[${relVar}:\`${relName}\`]->(t:\`${toDef.labels[0]}\`)\nWITH ${withParts.join(", ")}, ${collected} AS ${idsAlias}`,
    );
    withParts.push(idsAlias);
    edges.push({
      propName: relName,
      idsAlias,
      cardinality: rel.cardinality,
      relProps,
      load: loaderFactory(rel.to),
    });
  }

  // Auto-reverse: every plain rel whose `to` is this node is exposed inbound
  // under `_reverse`, keyed by the relation name, always as a `0..n` collection.
  for (const [relName, rel] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    if (isModuleRef(rel) || rel.to !== nodeKey) continue;
    const fromDef = schema.nodes[rel.from] as NodeDef;
    const idsAlias = `ids${i++}`;
    const relProps = rel.relProperties;
    const collected = relProps
      ? `collect(DISTINCT {id: t.id, rel: properties(r${idsAlias})})`
      : `collect(DISTINCT t.id)`;
    const relVar = relProps ? `r${idsAlias}` : "";
    matches.push(
      `OPTIONAL MATCH (n)<-[${relVar}:\`${relName}\`]-(t:\`${fromDef.labels[0]}\`)\nWITH ${withParts.join(", ")}, ${collected} AS ${idsAlias}`,
    );
    withParts.push(idsAlias);
    edges.push({
      propName: relName,
      idsAlias,
      cardinality: "0..n",
      relProps,
      reverse: true,
      load: loaderFactory(rel.from),
    });
  }

  // Outbound cross-module references from this node (carry a `module`, local
  // source). Inbound refs are handled in the loop below.
  for (const [relName, ref] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    if (!isModuleRef(ref) || isInboundModuleRef(ref) || ref.from !== nodeKey)
      continue;
    {
      const targetKey = ref.to ?? ref.module.schema.root;
      const targetDef = ref.module.schema.nodes[targetKey] as NodeDef | undefined;
      if (!targetDef) {
        throw new Error(
          `Module ref "${relName}" targets unknown node "${targetKey}" in module "${ref.module.schema.name}"`,
        );
      }
      const foreignTargetLabel = targetDef.labels[0];
      const idsAlias = `ids${i++}`;
      const relProps = ref.relProperties;
      const collected = relProps
        ? `collect(DISTINCT {id: t.id, rel: properties(r${idsAlias})})`
        : `collect(DISTINCT t.id)`;
      const relVar = relProps ? `r${idsAlias}` : "";
      matches.push(
        `OPTIONAL MATCH (n)-[${relVar}:\`${relName}\`]->(t:\`${foreignTargetLabel}\`)\nWITH ${withParts.join(", ")}, ${collected} AS ${idsAlias}`,
      );
      withParts.push(idsAlias);
      edges.push({
        propName: relName,
        idsAlias,
        cardinality: ref.cardinality,
        relProps,
        load: makeModuleLoader(ref.module, targetKey),
      });
    }
  }

  // Inbound cross-module references onto this node: a foreign module's node
  // points INTO us. Matched in the reverse direction and exposed under
  // `_reverse`, keyed by the relation name (always a `0..n` collection, like
  // auto-reverse). The source is loaded by the foreign module's own loader.
  for (const [relName, ref] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    if (!isInboundModuleRef(ref) || ref.to !== nodeKey) continue;
    const sourceDef = ref.module.schema.nodes[ref.from] as NodeDef | undefined;
    if (!sourceDef) {
      throw new Error(
        `Inbound module ref "${relName}" originates from unknown node "${ref.from}" in module "${ref.module.schema.name}"`,
      );
    }
    const foreignSourceLabel = sourceDef.labels[0];
    const idsAlias = `ids${i++}`;
    const relProps = ref.relProperties;
    const collected = relProps
      ? `collect(DISTINCT {id: t.id, rel: properties(r${idsAlias})})`
      : `collect(DISTINCT t.id)`;
    const relVar = relProps ? `r${idsAlias}` : "";
    matches.push(
      `OPTIONAL MATCH (n)<-[${relVar}:\`${relName}\`]-(t:\`${foreignSourceLabel}\`)\nWITH ${withParts.join(", ")}, ${collected} AS ${idsAlias}`,
    );
    withParts.push(idsAlias);
    edges.push({
      propName: relName,
      idsAlias,
      cardinality: "0..n",
      relProps,
      reverse: true,
      load: makeModuleLoader(ref.module, ref.from),
    });
  }

  const returnParts = ["n", ...edges.map((e) => e.idsAlias)];
  const cypher = [
    `MATCH (n:\`${label}\` {id: $id})`,
    ...matches,
    `RETURN ${returnParts.join(", ")}`,
  ].join("\n");

  const plan: NodeQueryPlan = { cypher, edges };
  perSchema.set(nodeKey, plan);
  return plan;
}

function makeModuleLoader(
  module: AnyModule,
  targetKey: string,
): (session: GraphSession, cache: LazyCache, id: string) => Promise<any> {
  return (session, cache, id) => {
    // Defer to the foreign module's own lazy loader, sharing the cache
    // so cross-module cycles also collapse to a single object per id.
    // `targetKey` is the node *inside* that module the edge lands on
    // (its root unless the ref declares `to`).
    const lazyMod = module as AnyModule & {
      __loadLazyNode?: (
        session: GraphSession,
        cache: LazyCache,
        nodeKey: string,
        id: string,
      ) => Promise<any>;
    };
    if (!lazyMod.__loadLazyNode) {
      throw new Error(
        `Module "${module.schema.name}" does not support fromLazyGraph`,
      );
    }
    return lazyMod.__loadLazyNode(session, cache, targetKey, id);
  };
}

function wrapCardinality(items: any[], cardinality: Cardinality): any {
  switch (cardinality) {
    case "1":
    case "0..1":
      return items[0] ?? null;
    case "1..n":
    case "0..n":
    default:
      return items;
  }
}

/**
 * Per-placement wrapper that carries the edge's own properties under `_rel`
 * while delegating every other property to the underlying (cached) node.
 *
 * The descriptors are copied (not prototype-linked) so the wrapper stays
 * enumerable and debuggable; the edge getters' memoization lives in their
 * closures, so it is still shared across placements. Only this thin wrapper is
 * per-edge — the underlying node remains cached by id, preserving cycle safety.
 */
function attachRel(node: any, relMap: Record<string, any>): any {
  if (node === null || typeof node !== "object") return node;
  const wrapper: Record<string, any> = {};
  Object.defineProperties(wrapper, Object.getOwnPropertyDescriptors(node));
  Object.defineProperty(wrapper, "_rel", {
    value: relMap,
    enumerable: true,
    configurable: false,
    writable: false,
  });
  return wrapper;
}

function extractScalarProps(raw: any): Record<string, any> {
  const props = raw.properties ? { ...raw.properties } : { ...raw };
  const labels: string[] = raw.labels ?? [];
  return { _id: props.id, _labels: labels, ...props };
}

// ---- Public factory ----

export interface LazyLoader {
  /** Load the root node lazily, wrapped as a chain for fluent traversal. */
  fromLazyGraph(session: GraphSession, rootId: string): LazyChain<any>;
  /**
   * Internal entry point used by cross-module loaders so all modules
   * reached from a single top-level call share one identity cache.
   */
  __loadLazyNode(
    session: GraphSession,
    cache: LazyCache,
    nodeKey: string,
    id: string,
  ): Promise<any>;
}

export function createLazyLoader(schema: ModuleSchema<any, any>): LazyLoader {
  const loaderFactory = (
    nodeKey: string,
  ): ((session: GraphSession, cache: LazyCache, id: string) => Promise<any>) => {
    return (session, cache, id) => loadNode(session, cache, nodeKey, id);
  };

  async function loadNode(
    session: GraphSession,
    cache: LazyCache,
    nodeKey: string,
    id: string,
  ): Promise<any> {
    const cacheKey = `${schema.name}::${id}`;
    const existing = cache.get(cacheKey);
    if (existing) return existing;

    const promise = (async () => {
      const plan = getNodePlan(schema, nodeKey, loaderFactory);
      const result = await session.run(plan.cypher, { id });
      if (result.records.length === 0) {
        throw new Error(
          `No node "${nodeKey}" with id "${id}" found in module "${schema.name}"`,
        );
      }
      const record = result.records[0];
      const raw = record.get("n");
      if (!raw) {
        throw new Error(
          `No node "${nodeKey}" with id "${id}" found in module "${schema.name}"`,
        );
      }

      const base: Record<string, any> = extractScalarProps(raw);

      // Every node carries a `_reverse` map; inbound edges install onto it
      // rather than onto the node directly, so they never collide with a
      // forward rel or module ref of the same name.
      const reverseContainer: Record<string, any> = {};
      Object.defineProperty(base, "_reverse", {
        value: reverseContainer,
        enumerable: true,
        configurable: false,
        writable: false,
      });

      for (const edge of plan.edges) {
        const column: unknown[] = record.get(edge.idsAlias) ?? [];
        installLazyEdge(edge.reverse ? reverseContainer : base, edge, session, cache, column);
      }

      return base;
    })();

    cache.set(cacheKey, promise);
    return promise;
  }

  function installLazyEdge(
    target: Record<string, any>,
    edge: EdgePlan,
    session: GraphSession,
    cache: LazyCache,
    column: unknown[],
  ): void {
    // Plain edges: column is `string[]` of target ids.
    // Property-bearing edges: column is `{id, rel}[]`; each child is wrapped
    // with a per-placement `_rel` block carrying the edge's own properties.
    const entries = edge.relProps
      ? (column as Array<{ id: string; rel: Record<string, any> }>)
      : (column as string[]).map((id) => ({ id, rel: undefined }));

    let memo: LazyChain<any> | undefined;
    Object.defineProperty(target, edge.propName, {
      enumerable: true,
      configurable: false,
      get() {
        if (memo) return memo;
        memo = chain(
          (async () => {
            const children: any[] = [];
            // Sequential: a Neo4j session allows only one in-flight
            // auto-commit transaction at a time, and a single edge can
            // fan out to many siblings whose own loads issue queries.
            for (const entry of entries) {
              const node = await edge.load(session, cache, entry.id);
              children.push(
                entry.rel === undefined ? node : attachRel(node, entry.rel),
              );
            }
            return wrapCardinality(children, edge.cardinality);
          })(),
        );
        return memo;
      },
    });
  }

  return {
    fromLazyGraph(session: GraphSession, rootId: string): LazyChain<any> {
      const cache: LazyCache = new Map();
      return chain(loadNode(session, cache, schema.root, rootId));
    },
    __loadLazyNode(session, cache, nodeKey, id) {
      return loadNode(session, cache, nodeKey, id);
    },
  };
}
