import { defineModule } from "../../../core/module";
import type { InferModuleLazyHydrated } from "../../../core/lazy";
import { PersonModule } from "../../Person/person.module";
import { session } from "neo4j-driver";


export const VideoFileModule = defineModule({
    schema: {
        name: "VideoFileModule",
        root: "Document",

        include: { Person: PersonModule },

        nodes: {
            Document: {
                labels: ["Document", "VideoFile"],
                properties: {
                    id: "string",
                },
            },
            Chunk: {
                labels: ["Chunk"],
                properties: {
                    id: "string",
                    content: "string",
                },
            },
            Entity: {
                // Triplet extraction
                // labels: ["Entity", "<entity_type>"],
                // will be added later and use this node
                // as generic entity node
                labels: ["Entity"],
                properties: {
                    id: "string",
                }
            }
        },

        relationships: {
            HAS_CHUNK: {
                from: "Document",
                to: "Chunk",
                cardinality: "1..n",
                relProperties: {
                    index: "number",
                },
            },

            SPEAKS_IN: {
                from: "Person",
                to: "Chunk",
                cardinality: "0..n",
                relProperties: {
                    id: "string",
                    role: "string",
                },
            },

            MENTIONS: {
                from: "Chunk",
                to: "Entity",
                cardinality: "0..n",
            },

            // triplet extraction
            // relation extracted by LLM, e.g. "isAuthorOf"
            // *: { 
            //     from: "Entity",
            //     to: "Entity",
            //     relProperties: {
            //         sources: "string[]"
            //     }
            // }
        },
    },
    async onDelete(ctx, data) {
        ctx.mutate(
            `
            MATCH (d:Document {id: $documentId})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[rel]-(nbr:Entity)
            WHERE rel.sources IS NOT NULL AND $documentId IN rel.sources
            WITH d,
                collect(DISTINCT c) AS chunks,
                [x IN collect(DISTINCT e) + collect(DISTINCT nbr) WHERE x IS NOT NULL] AS candidates

            // (A) Rule A: strip this doc from relation sources; delete emptied relations.
            CALL {
            WITH candidates
            UNWIND candidates AS ent
            MATCH (ent)-[r]-(:Entity)
            WHERE r.sources IS NOT NULL AND $documentId IN r.sources
            WITH DISTINCT r
            SET r.sources = [s IN r.sources WHERE s <> $documentId]
            WITH collect(r) AS rels
            FOREACH (x IN [y IN rels WHERE size(y.sources) = 0] | DELETE x)
            }

            // (B) delete the chunks (DETACH clears HAS_CHUNK, :MENTIONS, SPEAKS_IN).
            CALL {
            WITH chunks
            UNWIND chunks AS ch
            DETACH DELETE ch
            }

            // (C) delete the root document.
            WITH d, candidates
            DETACH DELETE d

            // (D) Rules B + C: delete candidates that are now fully orphaned.
            WITH candidates
            UNWIND candidates AS ent
            WITH DISTINCT ent
            WHERE NOT (ent)--()
            DELETE ent
            `,
            { documentId: data.id },
        );
    }
})

export type VideoFileHydrated = InferModuleLazyHydrated<typeof VideoFileModule>;
