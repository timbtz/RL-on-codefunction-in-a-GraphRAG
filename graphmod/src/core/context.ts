// ============================================================
// context.ts — DeleteContext + ModuleInstance (live graph handle)
// ============================================================

import type { GraphSession, DeleteContext, AnyModule } from "./types.js";

// ---- DeleteContext: queues mutations for batch execution ----

export function createDeleteContext(session: GraphSession): DeleteContext {
  const ops: { cypher: string; params: Record<string, unknown> }[] = [];

  const ctx: DeleteContext = {
    session,

    unlinkAll(nodeId) {
      ops.push({
        cypher: `MATCH ({ id: $id })-[r]-() DELETE r`,
        params: { id: nodeId },
      });
    },

    deleteNode(nodeId) {
      ops.push({
        cypher: `MATCH (n {id: $id}) DETACH DELETE n`,
        params: { id: nodeId },
      });
    },

    async deleteModule(module: AnyModule, nodeId: string) {
      // Flush pending ops first so ordering is preserved
      await ctx.flush();
      // Delegate to the sub-module's own delete handler
      await module.delete(session, nodeId);
    },

    mutate(cypher, params) {
      ops.push({ cypher, params });
    },

    async flush() {
      if (ops.length === 0) return;
      await session.executeWrite(async (tx) => {
        for (const op of ops) {
          await tx.run(op.cypher, op.params);
        }
      });
      ops.length = 0;
    },
  };

  return ctx;
}

// ---- ModuleInstance: live handle with mutation queue ----

export interface ModuleInstance<THydrated> {
  data: THydrated;
  session: GraphSession;
  /** Queue a Cypher mutation — executes on commit() */
  mutate(cypher: string, params: Record<string, unknown>): void;
  /** Flush all queued mutations */
  commit(): Promise<void>;
  /** Re-hydrate data from the graph */
  refresh(): Promise<void>;
}

export function createModuleInstance<THydrated>(
  module: {
    fromLazyGraph: (s: GraphSession, id: string) => PromiseLike<THydrated>;
  },
  session: GraphSession,
  rootId: string,
  data: THydrated,
): ModuleInstance<THydrated> {
  const queue: { cypher: string; params: Record<string, unknown> }[] = [];

  return {
    data,
    session,

    mutate(cypher, params) {
      queue.push({ cypher, params });
    },

    async commit() {
      if (queue.length === 0) return;
      await session.executeWrite(async (tx) => {
        for (const { cypher, params } of queue) {
          await tx.run(cypher, params);
        }
      });
      queue.length = 0;
    },

    async refresh() {
      this.data = await module.fromLazyGraph(session, rootId);
    },
  };
}
