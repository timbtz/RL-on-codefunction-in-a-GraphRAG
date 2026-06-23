// ============================================================
// cypher.ts — Cypher traversal-query generation + record id helpers
// ============================================================

import type { ModuleSchema, NodeDef, RelOrModule } from "./types";
import { isModuleRef, isInboundModuleRef } from "./types";

// ---- Build a traversal query from root outward ----

export interface RelAlias {
  relAlias: string;
  nodeAlias: string;
  relName: string;       // Neo4j relationship type
  fromKey: string;       // schema node key on the "from" side
  toKey: string;         // schema node key on the node bound to nodeAlias
  toLabel: string;
  direction: "forward" | "module";
  // When true, the forward MATCH was flipped to anchor on the to-side because
  // the from-side wasn't reachable. `nodeAlias` binds to the from-node; the
  // anchored to-node is at `anchorAlias`.
  bindsToSource?: boolean;
  anchorAlias?: string;
}

export function buildTraversalQuery(schema: ModuleSchema<any, any>): {
  cypher: string;
  aliases: RelAlias[];
} {
  const rootDef = schema.nodes[schema.root] as NodeDef;
  const rootLabel = rootDef.labels[0];

  const aliases: RelAlias[] = [];
  const matches: string[] = [];
  const returnParts: string[] = ["root"];

  const anchorOf = (nodeKey: string): string | null => {
    if (nodeKey === schema.root) return "root";
    const found = aliases.find((a) => a.toKey === nodeKey);
    return found?.nodeAlias ?? null;
  };

  let i = 0;
  for (const [relName, rel] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    // Module refs are anchored in a second pass below, after all plain
    // relationships have introduced their nodes (preserves anchoring order).
    if (isModuleRef(rel)) continue;
    const rAlias = `r${i}`;
    const nAlias = `n${i}`;
    const toDef = schema.nodes[rel.to] as NodeDef;
    const toLabel = toDef.labels[0];
    const fromDef = schema.nodes[rel.from] as NodeDef;
    const fromLabel = fromDef.labels[0];

    // Forward: (from)-[r:REL]->(to). Prefer to anchor on the from-side. If
    // the from-node isn't yet bound, flip the syntactic direction and anchor
    // on the to-side instead, binding nodeAlias to the from-node.
    const fromAnchor = anchorOf(rel.from);
    const toAnchor = anchorOf(rel.to);

    if (fromAnchor) {
      matches.push(
        `OPTIONAL MATCH (${fromAnchor})-[${rAlias}:\`${relName}\`]->(${nAlias}:\`${toLabel}\`)`,
      );
      returnParts.push(rAlias, nAlias);
      aliases.push({
        relAlias: rAlias,
        nodeAlias: nAlias,
        relName,
        fromKey: rel.from,
        toKey: rel.to,
        toLabel,
        direction: "forward",
      });
    } else if (toAnchor) {
      matches.push(
        `OPTIONAL MATCH (${nAlias}:\`${fromLabel}\`)-[${rAlias}:\`${relName}\`]->(${toAnchor})`,
      );
      returnParts.push(rAlias, nAlias);
      aliases.push({
        relAlias: rAlias,
        nodeAlias: nAlias,
        relName,
        fromKey: rel.from,
        toKey: rel.from, // nodeAlias binds to the from-side node
        toLabel: fromLabel,
        direction: "forward",
        bindsToSource: true,
        anchorAlias: toAnchor,
      });
    } else {
      throw new Error(
        `Cannot anchor relationship "${relName}": neither "${rel.from}" nor "${rel.to}" is reachable from root`,
      );
    }
    i++;
    // Auto-reverse edges are unconstrained (`0..n`), so the validation
    // traversal never needs to query the inbound direction.
  }

  // Outbound module references (carry a `module`, local source). Inbound refs
  // land under `_reverse`, are unconstrained (`0..n`), and so are never queried
  // for cardinality validation.
  for (const [relName, ref] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    if (!isModuleRef(ref) || isInboundModuleRef(ref)) continue;
    {
      const rAlias = `r${i}`;
      const nAlias = `n${i}`;
      const foreignRootLabel = ref.module.schema.nodes[ref.module.schema.root].labels[0];
      const fromAlias = anchorOf(ref.from);
      if (!fromAlias) {
        throw new Error(
          `Cannot anchor module reference "${relName}": "${ref.from}" is not reachable from root`,
        );
      }

      matches.push(
        `OPTIONAL MATCH (${fromAlias})-[${rAlias}:\`${relName}\`]->(${nAlias}:\`${foreignRootLabel}\`)`,
      );
      returnParts.push(rAlias, nAlias);

      aliases.push({
        relAlias: rAlias,
        nodeAlias: nAlias,
        relName,
        fromKey: ref.from,
        toKey: `__module__${relName}`,
        toLabel: foreignRootLabel,
        direction: "module",
      });
      i++;
    }
  }

  const cypher = [
    `MATCH (root:\`${rootLabel}\` {id: $rootId})`,
    ...matches,
    `RETURN ${returnParts.join(", ")}`,
  ].join("\n");

  return { cypher, aliases };
}

// ---- Helpers ----

export function extractId(raw: any): string {
  const props = raw.properties ?? raw;
  return props.id;
}

export function resolveFromId(
  record: any,
  alias: RelAlias,
  allAliases: RelAlias[],
  schema: ModuleSchema<any, any>,
): string | null {
  if (alias.fromKey === schema.root) {
    const root = record.get("root");
    return root ? extractId(root) : null;
  }

  const parentAlias = allAliases.find((a) => a.toKey === alias.fromKey);
  if (!parentAlias) return null;
  const parentRaw = record.get(parentAlias.nodeAlias);
  return parentRaw ? extractId(parentRaw) : null;
}
