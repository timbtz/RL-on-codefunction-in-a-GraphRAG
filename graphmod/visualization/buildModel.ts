// ============================================================
// buildModel.ts — Introspect module schemas into a flat graph model
// ============================================================
//
// Everything needed to draw a module lives in `module.schema`: its nodes
// (incl. composed `include` aliases, which `defineModule` merges into
// `schema.nodes`), and its `relationships` map — plain intra-module edges
// plus outbound/inbound cross-module refs. This file turns one-or-more
// modules into a serializable `GraphModel` the renderer can consume, with no
// database access.

import {
  type AnyModule,
  type NodeDef,
  type PropType,
  type Cardinality,
  type ModuleRef,
  type InboundModuleRef,
  type RelationshipDef,
  type RelOrModule,
  isModuleRef,
  isInboundModuleRef,
} from "../src/core/types.js";

export interface PropInfo {
  name: string;
  type: PropType;
}

/** The kind of edge, which drives how it is styled. */
export type EdgeKind = "intra" | "outbound" | "inbound" | "include";

export interface NodeInfo {
  /** Stable id: `${moduleName}::${key}`. */
  id: string;
  moduleName: string;
  /** Node key within the schema (also the `from`/`to` key in relationships). */
  key: string;
  labels: string[];
  properties: PropInfo[];
  isRoot: boolean;
  /** Set when this node is an `include` alias — names the origin module. */
  includedFrom?: string;
}

export interface EdgeInfo {
  id: string;
  source: string;
  target: string;
  /** Relationship name (the key in the `relationships` map). */
  name: string;
  cardinality: Cardinality;
  kind: EdgeKind;
  relProperties: PropInfo[];
}

export interface ModuleInfo {
  name: string;
  root: string;
  /** Names of modules this one references (via include or cross-module refs). */
}

export interface GraphModel {
  modules: ModuleInfo[];
  nodes: NodeInfo[];
  edges: EdgeInfo[];
}

const nodeId = (moduleName: string, key: string) => `${moduleName}::${key}`;

function toProps(rec: Record<string, PropType> | undefined): PropInfo[] {
  if (!rec) return [];
  return Object.entries(rec).map(([name, type]) => ({ name, type }));
}

/**
 * Walk `include` and cross-module refs from the seed modules to collect every
 * module that participates in the graph, so cross-module edges always land on
 * a real node. Keyed by `schema.name` (assumed unique).
 */
function discoverModules(seed: AnyModule[]): Map<string, AnyModule> {
  const found = new Map<string, AnyModule>();
  const queue = [...seed];

  while (queue.length) {
    const mod = queue.shift()!;
    const name = mod.schema.name;
    if (found.has(name)) continue;
    found.set(name, mod);

    const { include, relationships } = mod.schema;

    for (const inc of Object.values(include ?? {})) {
      if (inc && !found.has(inc.schema.name)) queue.push(inc);
    }
    for (const def of Object.values(
      (relationships ?? {}) as Record<string, RelOrModule>,
    )) {
      if (isModuleRef(def) && def.module && !found.has(def.module.schema.name)) {
        queue.push(def.module);
      }
    }
  }
  return found;
}

/**
 * Build a flat, serializable graph model from one or more modules. Any module
 * referenced (via `include` or a cross-module ref) but not passed in is
 * auto-discovered and rendered too, so every edge endpoint resolves.
 */
export function buildGraphModel(modules: AnyModule[]): GraphModel {
  const registry = discoverModules(modules);

  const moduleInfos: ModuleInfo[] = [];
  const nodes: NodeInfo[] = [];
  const edges: EdgeInfo[] = [];
  const nodeIndex = new Set<string>();

  // ---- Nodes ----
  for (const mod of registry.values()) {
    const { name, root, nodes: nodeDefs, include } = mod.schema;
    moduleInfos.push({ name, root });

    // Reverse-map: which alias keys came from an included module.
    const includedFrom: Record<string, string> = {};
    for (const [alias, inc] of Object.entries(include ?? {})) {
      if (inc) includedFrom[alias] = inc.schema.name;
    }

    for (const [key, def] of Object.entries(
      nodeDefs as Record<string, NodeDef>,
    )) {
      const id = nodeId(name, key);
      nodeIndex.add(id);
      nodes.push({
        id,
        moduleName: name,
        key,
        labels: def.labels,
        properties: toProps(def.properties),
        isRoot: key === root,
        includedFrom: includedFrom[key],
      });
    }
  }

  // Connect an endpoint, tolerating refs to nodes/modules outside the set by
  // synthesizing a placeholder so the edge is never dropped silently.
  const ensure = (moduleName: string, key: string) => {
    const id = nodeId(moduleName, key);
    if (!nodeIndex.has(id)) {
      nodeIndex.add(id);
      // Place the placeholder in its own module if unknown, else its module.
      const exists = registry.has(moduleName);
      if (!exists) moduleInfos.push({ name: moduleName, root: key });
      nodes.push({
        id,
        moduleName,
        key,
        labels: [key],
        properties: [],
        isRoot: false,
      });
    }
    return id;
  };

  // ---- Edges ----
  for (const mod of registry.values()) {
    const { name, include, relationships } = mod.schema;

    // include → dotted edge from the local alias to the origin module's root.
    for (const [alias, inc] of Object.entries(include ?? {})) {
      if (!inc) continue;
      const source = nodeId(name, alias);
      const target = ensure(inc.schema.name, inc.schema.root);
      if (!nodeIndex.has(source)) continue;
      edges.push({
        id: `${name}::include::${alias}`,
        source,
        target,
        name: "includes",
        cardinality: "1",
        kind: "include",
        relProperties: [],
      });
    }

    for (const [relName, def] of Object.entries(
      (relationships ?? {}) as Record<string, RelOrModule>,
    )) {
      if (!isModuleRef(def)) {
        // Plain intra-module relationship.
        const r = def as RelationshipDef;
        edges.push({
          id: `${name}::${relName}`,
          source: ensure(name, r.from),
          target: ensure(name, r.to),
          name: relName,
          cardinality: r.cardinality,
          kind: "intra",
          relProperties: toProps(r.relProperties),
        });
      } else if (isInboundModuleRef(def)) {
        // Foreign source node → local node.
        const r = def as InboundModuleRef;
        const source = ensure(r.module.schema.name, r.from);
        const target = ensure(name, r.to);
        edges.push({
          id: `${name}::${relName}`,
          source,
          target,
          name: relName,
          cardinality: r.cardinality,
          kind: "inbound",
          relProperties: toProps(r.relProperties),
        });
      } else {
        // Outbound: local node → foreign module node (`to` or its root).
        const r = def as ModuleRef;
        const targetKey = r.to ?? r.module.schema.root;
        const source = ensure(name, r.from);
        const target = ensure(r.module.schema.name, targetKey);
        edges.push({
          id: `${name}::${relName}`,
          source,
          target,
          name: relName,
          cardinality: r.cardinality,
          kind: "outbound",
          relProperties: toProps(r.relProperties),
        });
      }
    }
  }

  return { modules: moduleInfos, nodes, edges };
}
