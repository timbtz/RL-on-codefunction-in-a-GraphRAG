import { defineModule } from "../../../core/module";

// Shared text node — the universal searchable unit across all three sources.
// Every Document / Message / Ticket owns one or more Chunks via HAS_CHUNK; the
// fulltext index is built over Chunk.content so the seed search has a single
// node type to seed from. MENTIONS edges (Chunk -> Entity) hang off chunks in
// each content module. Kept shared so cross-source entity resolution and the
// LLM-extraction lever operate on one chunk layer.
export const ChunkModule = defineModule({
    schema: {
        name: "ChunkModule",
        root: "Chunk",

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

        relationships: {},
    },
    async onDelete(ctx, data) {
        ctx.unlinkAll(data.id);
        ctx.deleteNode(data.id);
    },
});
