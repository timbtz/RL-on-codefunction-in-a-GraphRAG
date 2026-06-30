// export-schema.ts — dump the Corpus modules' allowed vocabulary to schema.json.
//
// The LLM extractor (extract.ts) and the read-only attribution probe
// (graph_inspect.py) need to know which node labels and relationship triplets
// are legal, so an LLM-proposed entity/relation can be pruned if off-schema.
//
//   npx tsx graphmod/scripts/export-schema.ts            # writes graphmod/schema.json
//   npx tsx graphmod/scripts/export-schema.ts > out.json # also prints to stdout
//
// No DB connection required — pure schema introspection.

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { CORPUS_MODULES } from "../src/modules/Corpus/index";
import type { NodeDef, RelationshipDef } from "../src/core/types";

interface ExportedSchema {
    labels: string[];
    nodeProps: Record<string, Record<string, string>>; // primaryLabel -> {prop: PropType}
    relationships: { type: string; from: string; to: string }[];
}

function isPlainRel(rel: unknown): rel is RelationshipDef {
    return (
        typeof rel === "object" &&
        rel !== null &&
        "from" in rel &&
        "to" in rel &&
        typeof (rel as RelationshipDef).from === "string" &&
        typeof (rel as RelationshipDef).to === "string"
    );
}

function build(): ExportedSchema {
    // name (node key / include alias) -> primary label + props
    const nodeByName = new Map<string, { label: string; props: Record<string, string> }>();
    const labels = new Set<string>();
    const rels = new Map<string, { type: string; from: string; to: string }>();

    for (const mod of CORPUS_MODULES) {
        const nodes = mod.schema.nodes as Record<string, NodeDef>;
        for (const [name, def] of Object.entries(nodes)) {
            const primary = def.labels[0];
            nodeByName.set(name, {
                label: primary,
                props: def.properties as Record<string, string>,
            });
            for (const l of def.labels) labels.add(l);
        }
    }

    const labelOf = (name: string): string =>
        nodeByName.get(name)?.label ?? name; // fall back to the alias itself

    for (const mod of CORPUS_MODULES) {
        const relationships = mod.schema.relationships as Record<string, unknown>;
        for (const [type, rel] of Object.entries(relationships)) {
            if (!isPlainRel(rel)) continue; // skip ModuleRef / InboundModuleRef
            const from = labelOf(rel.from);
            const to = labelOf(rel.to);
            rels.set(`${type}:${from}->${to}`, { type, from, to });
        }
    }

    const nodeProps: Record<string, Record<string, string>> = {};
    for (const { label, props } of nodeByName.values()) {
        nodeProps[label] = { ...(nodeProps[label] ?? {}), ...props };
    }

    return {
        labels: [...labels].sort(),
        nodeProps,
        relationships: [...rels.values()].sort((a, b) =>
            `${a.type}${a.from}${a.to}`.localeCompare(`${b.type}${b.from}${b.to}`),
        ),
    };
}

const schema = build();
const json = JSON.stringify(schema, null, 2);

// Always write the canonical file next to graphmod/.
const here = dirname(fileURLToPath(import.meta.url)); // graphmod/scripts
const outPath = join(here, "..", "schema.json");
writeFileSync(outPath, json + "\n");
process.stderr.write(
    `export-schema: ${schema.labels.length} labels, ${schema.relationships.length} relationship triplets -> ${outPath}\n`,
);

// Also print to stdout so callers can redirect.
process.stdout.write(json + "\n");
