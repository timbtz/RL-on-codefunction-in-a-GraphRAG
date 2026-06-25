// ============================================================
// validate.ts — Runtime cardinality validation
// ============================================================

import type {
  ModuleSchema,
  NodeDef,
  RelationshipDef,
  ModuleRef,
  RelOrModule,
  Cardinality,
  PropType,
  GraphTransaction,
} from "./types.js";
import { isModuleRef, isInboundModuleRef } from "./types.js";
import { buildTraversalQuery, extractId, resolveFromId } from "./cypher.js";

export interface ValidationError {
  path: string;
  relationship: string;
  expected: Cardinality | PropType | "required";
  actual: number | "missing" | "present" | string;
  message: string;
}

function isOptionalProp(t: PropType): boolean {
  return t.endsWith("?");
}

/** Neo4j integer object — duck-type so we don't need a driver import here. */
function isNeo4jInteger(v: unknown): boolean {
  return (
    typeof v === "object" &&
    v !== null &&
    typeof (v as { low?: unknown }).low === "number" &&
    typeof (v as { high?: unknown }).high === "number"
  );
}

/** Neo4j temporal (`DateTime`/`Date`) — duck-typed by its year/month/day fields. */
function isNeo4jTemporal(v: unknown): boolean {
  return (
    typeof v === "object" &&
    v !== null &&
    "year" in v &&
    "month" in v &&
    "day" in v
  );
}

/** Returns null if value matches the declared PropType, else a human label of the actual type. */
function checkPropType(value: unknown, propType: PropType): string | null {
  const base = propType.replace(/\?$/, "") as
    | "string"
    | "number"
    | "boolean"
    | "datetime"
    | "string[]"
    | "number[]";

  const actual = (): string => {
    if (Array.isArray(value)) {
      if (value.length === 0) return "unknown[]";
      const first = value[0];
      if (typeof first === "string") return "string[]";
      if (typeof first === "number" || isNeo4jInteger(first)) return "number[]";
      return `${typeof first}[]`;
    }
    if (isNeo4jTemporal(value)) return "datetime";
    if (isNeo4jInteger(value)) return "number";
    return typeof value;
  };

  switch (base) {
    case "string":
      return typeof value === "string" ? null : actual();
    case "number":
      return typeof value === "number" || isNeo4jInteger(value) ? null : actual();
    case "boolean":
      return typeof value === "boolean" ? null : actual();
    case "datetime":
      return isNeo4jTemporal(value) ? null : actual();
    case "string[]":
      return Array.isArray(value) && value.every((v) => typeof v === "string")
        ? null
        : actual();
    case "number[]":
      return Array.isArray(value) &&
        value.every((v) => typeof v === "number" || isNeo4jInteger(v))
        ? null
        : actual();
  }
}

export class SchemaValidationError extends Error {
  readonly errors: ValidationError[];
  constructor(moduleName: string, errors: ValidationError[]) {
    super(
      `Module "${moduleName}" failed schema validation:\n` +
        errors.map((e) => `  - ${e.message}`).join("\n"),
    );
    this.name = "SchemaValidationError";
    this.errors = errors;
  }
}

export function validateCardinality(
  schema: ModuleSchema<any, any>,
  graph: any,
  path = "root",
): ValidationError[] {
  const errors: ValidationError[] = [];

  // Check relationships and module references (both live in `relationships`).
  for (const [relName, rel] of Object.entries(
    schema.relationships as Record<string, RelOrModule>,
  )) {
    // Inbound module refs are reached under `_reverse` and are unconstrained
    // (`0..n`) — nothing to validate here.
    if (isInboundModuleRef(rel)) continue;
    const moduleRef = isModuleRef(rel);
    // Plain rels below the root are only checked when we've recursed onto their
    // `from` node; module refs are checked wherever they appear.
    if (!moduleRef && rel.from !== schema.root && path === "root") continue;

    const value = graph[relName];
    const count = Array.isArray(value) ? value.length : value ? 1 : 0;

    const invalid =
      (rel.cardinality === "1" && count !== 1) ||
      (rel.cardinality === "0..1" && count > 1) ||
      (rel.cardinality === "1..n" && count < 1);

    if (invalid) {
      errors.push({
        path,
        relationship: relName,
        expected: rel.cardinality,
        actual: count,
        message: moduleRef
          ? `${path}.${relName} (module ref): expected ${rel.cardinality}, got ${count}`
          : `${path}.${relName}: expected ${rel.cardinality}, got ${count}`,
      });
    }

    // Recurse into children of plain relationships only. Module refs cross into
    // a foreign module and are validated only for top-level cardinality.
    if (moduleRef) continue;
    const children = Array.isArray(value) ? value : value ? [value] : [];
    for (const child of children) {
      errors.push(
        ...validateCardinality(
          { ...schema, root: rel.to } as any,
          child,
          `${path}.${relName}`,
        ),
      );
    }
  }

  return errors;
}

/**
 * Run the module's traversal query *inside* the given transaction, count
 * relationships per node, and validate cardinality against the schema.
 *
 * Throws `SchemaValidationError` on mismatch. When called from within a
 * `GraphSession.executeWrite` block, the thrown error aborts the transaction
 * and rolls back all uncommitted writes.
 *
 * Note: module references (entries in `schema.relationships` carrying a
 * `module`) are only validated for their top-level cardinality — sub-modules
 * are not recursively re-validated here.
 */
export async function assertValidInTx(
  tx: GraphTransaction,
  schema: ModuleSchema<any, any>,
  rootId: string,
): Promise<void> {
  const { cypher, aliases } = buildTraversalQuery(schema);
  const result = await tx.run(cypher, { rootId });

  if (result.records.length === 0) {
    throw new SchemaValidationError(schema.name, [
      {
        path: "root",
        relationship: "_root",
        expected: "1",
        actual: 0,
        message: `root node "${rootId}" not found`,
      },
    ]);
  }

  // nodeId → schema node key (e.g., "Document", "Chunk")
  const nodeKind = new Map<string, string>();
  // nodeId → raw property bag (for required-property checks)
  const nodeProps = new Map<string, Record<string, unknown>>();
  // edgeKey → fromId → Set<toId>
  // edgeKey is the relation name (forward + module). Auto-reverse edges are
  // unconstrained (`0..n`), so they are neither queried nor counted here.
  const edges = new Map<string, Map<string, Set<string>>>();
  // Edge-property instances to type-check, deduped by `${relName}|${from}|${to}`.
  const edgeProps = new Map<
    string,
    { relName: string; fromId: string; toId: string; props: Record<string, unknown> }
  >();

  function recordEdge(key: string, fromId: string, toId: string) {
    let m = edges.get(key);
    if (!m) edges.set(key, (m = new Map()));
    let s = m.get(fromId);
    if (!s) m.set(fromId, (s = new Set()));
    s.add(toId);
  }

  const extractProps = (raw: any): Record<string, unknown> =>
    raw?.properties ? { ...raw.properties } : { ...raw };

  for (const record of result.records) {
    const rootRaw = record.get("root");
    if (rootRaw) {
      const id = extractId(rootRaw);
      if (!nodeKind.has(id)) {
        nodeKind.set(id, schema.root);
        nodeProps.set(id, extractProps(rootRaw));
      }
    }

    for (const alias of aliases) {
      const raw = record.get(alias.nodeAlias);
      if (!raw) continue;

      const id = extractId(raw);
      // Module-ref nodes have alias.toKey = `__module__${relName}` — keep that
      // marker so we don't accidentally validate them as schema nodes.
      if (!nodeKind.has(id)) {
        nodeKind.set(id, alias.toKey);
        nodeProps.set(id, extractProps(raw));
      }

      let edgeFromId: string | null;
      let edgeToId: string;
      if (alias.bindsToSource) {
        const anchorRaw = record.get(alias.anchorAlias!);
        if (!anchorRaw) continue;
        edgeFromId = id;
        edgeToId = extractId(anchorRaw);
      } else {
        edgeFromId = resolveFromId(record, alias, aliases, schema);
        edgeToId = id;
      }
      if (!edgeFromId) continue;

      recordEdge(alias.relName, edgeFromId, edgeToId);

      // Capture the edge's own properties for any rel/module-ref that declares
      // them. Plain rels and module refs both live in `relationships`.
      const def = (schema.relationships as Record<string, RelOrModule>)[
        alias.relName
      ];
      const propsDef = def?.relProperties;
      if (propsDef && alias.relAlias) {
        const relRaw = record.get(alias.relAlias);
        if (relRaw) {
          // The relationship always points from→to; the dedup key is
          // (relName, source, target).
          const key = `${alias.relName}|${edgeFromId}|${edgeToId}`;
          if (!edgeProps.has(key)) {
            edgeProps.set(key, {
              relName: alias.relName,
              fromId: edgeFromId,
              toId: edgeToId,
              props: extractProps(relRaw),
            });
          }
        }
      }
    }
  }

  const errors: ValidationError[] = [];

  // Type-check a property bag (node or edge) against its declared types.
  const checkProps = (
    props: Record<string, unknown>,
    defs: Record<string, PropType>,
    path: string,
  ) => {
    for (const [propName, propType] of Object.entries(defs)) {
      const value = props[propName];
      const missing = value === undefined || value === null;

      if (missing) {
        if (isOptionalProp(propType)) continue;
        errors.push({
          path,
          relationship: propName,
          expected: "required",
          actual: "missing",
          message: `${path}.${propName}: required ${propType} property is missing`,
        });
        continue;
      }

      const actualType = checkPropType(value, propType);
      if (actualType !== null) {
        errors.push({
          path,
          relationship: propName,
          expected: propType,
          actual: actualType,
          message: `${path}.${propName}: expected ${propType}, got ${actualType}`,
        });
      }
    }
  };

  const check = (
    path: string,
    relName: string,
    expected: Cardinality,
    actual: number,
  ) => {
    const invalid =
      (expected === "1" && actual !== 1) ||
      (expected === "0..1" && actual > 1) ||
      (expected === "1..n" && actual < 1);
    if (invalid) {
      errors.push({
        path,
        relationship: relName,
        expected,
        actual,
        message: `${path}.${relName}: expected ${expected}, got ${actual}`,
      });
    }
  };

  for (const [nodeId, kind] of nodeKind) {
    const path = kind === schema.root ? "root" : kind;

    // Required-property check (skip module-ref placeholder nodes)
    const nodeDef = (schema.nodes as Record<string, NodeDef>)[kind];
    if (nodeDef) {
      checkProps(nodeProps.get(nodeId) ?? {}, nodeDef.properties, path);
    }

    // Forward + module-ref cardinality (both keyed by relName). Auto-reverse
    // edges are `0..n` and unconstrained, so there is nothing to check inbound.
    for (const [relName, rel] of Object.entries(
      schema.relationships as Record<string, RelOrModule>,
    )) {
      if (rel.from === kind) {
        const cnt = edges.get(relName)?.get(nodeId)?.size ?? 0;
        check(path, relName, rel.cardinality, cnt);
      }
    }
  }

  // Validate edge (relationship + module-ref) properties.
  for (const { relName, fromId, toId, props } of edgeProps.values()) {
    const def = (schema.relationships as Record<string, RelOrModule>)[relName];
    const defs = def?.relProperties;
    if (!defs) continue;
    const modDef = def && isModuleRef(def) ? (def as ModuleRef) : undefined;
    const relDef = def && !modDef ? (def as RelationshipDef) : undefined;
    const fromKind = nodeKind.get(fromId) ?? def?.from ?? "?";
    // For module refs the target is the foreign module's root, so prefer the
    // foreign module name over the `__module__*` placeholder kind.
    const toKind = modDef
      ? modDef.module.schema.name
      : nodeKind.get(toId) ?? relDef?.to ?? "?";
    checkProps(props, defs, `${fromKind}-[${relName}]->${toKind}`);
  }

  if (errors.length > 0) {
    throw new SchemaValidationError(schema.name, errors);
  }
}
