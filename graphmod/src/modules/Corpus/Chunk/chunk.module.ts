import { defineModule } from "../../../core/module";
import { EntityModule } from "../Entity/entity.module";

// Shared text node — the universal searchable unit across all three sources.
// Every Document / Message / Ticket owns one or more Chunks via HAS_CHUNK; the
// fulltext index is built over Chunk.content so the seed search has a single
// node type to seed from. Kept shared so cross-source entity resolution and the
// LLM-extraction lever operate on one chunk layer.
//
// MENTIONS fans out from Chunk to Entity, Person, Component AND Ticket (regex +
// gazetteer, see extract.ts collectMentions) regardless of source. Relationship
// keys are unique per module, so the four triplets are declared one-per-module
// and unioned by export-schema: Chunk->Entity here (the base chunk layer),
// Chunk->Person in DocModule, Chunk->Component in MessageModule and
// Chunk->Ticket in TicketModule.
export const ChunkModule = defineModule({
    schema: {
        name: "ChunkModule",
        root: "Chunk",

        include: {
            Entity: EntityModule,
        },

        nodes: {
            Chunk: {
                labels: ["Chunk"],
                properties: {
                    id: "string",
                    content: "string",
                    // origin source kind: "doc" | "chat" | "jira" (provenance for attribution)
                    source: "string?",
                },
            },
        },

        relationships: {
            MENTIONS: {
                from: "Chunk",
                to: "Entity",
                cardinality: "0..n",
            },
        },
    },
    async onDelete(ctx, data) {
        ctx.unlinkAll(data.id);
        ctx.deleteNode(data.id);
    },
});
