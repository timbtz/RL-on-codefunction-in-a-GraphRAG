// ============================================================
// module.ts — defineModule: schema + lazy hydration + delete
// ============================================================

import type {
  NodeDef,
  RelOrModule,
  ModuleSchema,
  HydratedNode,
  DeleteHandler,
  GraphSession,
  GraphTransaction,
  AnyModule,
} from "./types.js";
import { createDeleteContext } from "./context.js";
import {
  validateCardinality,
  assertValidInTx,
  type ValidationError,
} from "./validate.js";
import { createLazyLoader, type LazyHydratedNode, type LazyChain } from "./lazy.js";

// ---- Module object (what defineModule returns) ----

export interface Module<
  TNodes extends Record<string, NodeDef>,
  TRels extends Record<string, RelOrModule<TNodes>>,
  TRoot extends keyof TNodes & string,
> extends AnyModule {
  schema: ModuleSchema<TNodes, TRels>;
  onDelete: DeleteHandler<LazyHydratedNode<TNodes, TRels, TRoot>>;

  /**
   * Phantom type carrier — never populated at runtime. Maps each node key to
   * its lazily-hydrated shape so a cross-module `ModuleRef` declaring `to`
   * can resolve the target sub-node's type (see `ResolveModuleLazyNode`).
   */
  readonly __lazyNodes?: {
    [K in keyof TNodes & string]: LazyHydratedNode<TNodes, TRels, K>;
  };

  /**
   * Hydrate the subgraph from a root node ID. Every relationship and module
   * reference is exposed as a memoized chainable `LazyChain`-valued property that
   * fetches on first access. Cycles terminate because nodes are cached by
   * id within a single call, so revisiting a node returns the same JS
   * object. The returned root is itself a chain — `await` it to get the
   * resolved node, or keep chaining (`fromLazyGraph(s, id).A.at(0).B`).
   */
  fromLazyGraph(
    session: GraphSession,
    rootId: string,
  ): LazyChain<LazyHydratedNode<TNodes, TRels, TRoot>>;

  /** Delete the module's subgraph using its onDelete handler */
  delete(session: GraphSession, rootId: string): Promise<void>;

  /** Validate cardinality constraints on a hydrated graph */
  validate(
    data: HydratedNode<TNodes, TRels, TRoot>,
  ): ValidationError[];

  /**
   * Validate the subgraph by reading through the given transaction and
   * checking cardinality against the schema. Throws `SchemaValidationError`
   * on mismatch — if called inside `session.executeWrite`, this aborts the
   * transaction and rolls back all uncommitted writes.
   */
  assertValid(tx: GraphTransaction, rootId: string): Promise<void>;
}

// ---- Include / compose ----
//
// `include: { alias: ForeignModule }` merges each foreign module's *root* node
// def into the local node set under `alias`. After merging, plain relationships
// may reference `alias` as an ordinary local node in either direction. This map
// resolves the merged node type so `TRels` can be checked against the composed
// node set and the hydrated shapes pick up the included nodes.

type IncludedNodes<TInclude extends Record<string, AnyModule>> = {
  [K in keyof TInclude]: TInclude[K] extends Module<infer N, any, infer R>
    ? R extends keyof N
      ? N[R]
      : NodeDef
    : NodeDef;
};

// ---- Factory ----

export function defineModule<
  const TRoot extends string,
  const TLocalNodes extends Record<string, NodeDef> & Record<TRoot, NodeDef>,
  const TInclude extends Record<string, AnyModule> = {},
  const TRels extends Record<
    string,
    RelOrModule<TLocalNodes & IncludedNodes<TInclude>>
  > = Record<string, RelOrModule<TLocalNodes & IncludedNodes<TInclude>>>,
>(config: {
  schema: {
    name: string;
    root: TRoot;
    nodes: TLocalNodes;
    include?: TInclude;
    relationships: TRels;
  };
  onDelete: DeleteHandler<
    LazyHydratedNode<TLocalNodes & IncludedNodes<TInclude>, TRels, TRoot>
  >;
}): Module<TLocalNodes & IncludedNodes<TInclude>, TRels, TRoot> {
  const { schema, onDelete } = config;

  type TNodes = TLocalNodes & IncludedNodes<TInclude>;
  type Hydrated = HydratedNode<TNodes, TRels, TRoot>;
  type LazyHydrated = LazyHydratedNode<TNodes, TRels, TRoot>;

  // Merge included modules' root node defs into the local node set, so the
  // lazy loader, validator, and Cypher builder all see the composed schema.
  const mergedNodes: Record<string, NodeDef> = {
    ...(schema.nodes as Record<string, NodeDef>),
  };
  if (schema.include) {
    for (const [alias, inc] of Object.entries(schema.include)) {
      const rootKey = inc.schema.root;
      const rootDef = inc.schema.nodes[rootKey] as NodeDef | undefined;
      if (!rootDef) {
        throw new Error(
          `Cannot include module "${inc.schema.name}": its root node "${rootKey}" is not defined`,
        );
      }
      if (!(alias in mergedNodes)) mergedNodes[alias] = rootDef;
    }
  }
  const runtimeSchema = {
    ...schema,
    nodes: mergedNodes,
  } as ModuleSchema<any, any>;

  const lazy = createLazyLoader(runtimeSchema);

  const mod: Module<TNodes, TRels, TRoot> = {
    schema: runtimeSchema as ModuleSchema<TNodes, TRels>,
    onDelete,

    fromLazyGraph(session: GraphSession, rootId: string): LazyChain<LazyHydrated> {
      return lazy.fromLazyGraph(session, rootId) as LazyChain<LazyHydrated>;
    },

    async delete(session: GraphSession, rootId: string): Promise<void> {
      const data = (await mod.fromLazyGraph(session, rootId)) as LazyHydrated;
      const ctx = createDeleteContext(session);
      await onDelete(ctx, data);
      await ctx.flush();
    },

    validate(data: Hydrated): ValidationError[] {
      return validateCardinality(runtimeSchema, data);
    },

    async assertValid(tx: GraphTransaction, rootId: string): Promise<void> {
      await assertValidInTx(tx, runtimeSchema, rootId);
    },
  };

  // Reserved hook so cross-module lazy loaders can share the per-call
  // identity cache of the originating fromLazyGraph invocation.
  (mod as unknown as { __loadLazyNode: typeof lazy.__loadLazyNode }).__loadLazyNode =
    lazy.__loadLazyNode;

  return mod;
}
