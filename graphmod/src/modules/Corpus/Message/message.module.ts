import { defineModule } from "../../../core/module";
import { PersonModule } from "../../Person/person.module";
import { ComponentModule } from "../Component/component.module";
import { EntityModule } from "../Entity/entity.module";
import { ChunkModule } from "../Chunk/chunk.module";
import { DocModule } from "../Doc/doc.module";

// Chat message source. Structured fields give SENT_BY (-> Person) and REFERENCES
// (-> Document); regex/gazetteer over the message text gives MENTIONS
// (Chunk -> Entity); thread structure gives REPLY_TO. The implicit relations
// buried in free-text message bodies (e.g. "(tim)-[:REVIEWED]->(doc)") are
// exactly what rule-based extraction CANNOT recover — the seam the LLM lever
// fills in M4.
export const MessageModule = defineModule({
    schema: {
        name: "MessageModule",
        root: "Message",

        include: {
            Person: PersonModule,
            Component: ComponentModule,
            Entity: EntityModule,
            Chunk: ChunkModule,
            Document: DocModule,
        },

        nodes: {
            Message: {
                labels: ["Message"],
                properties: {
                    id: "string",
                    text: "string",
                    ts: "datetime?",
                    channel: "string?",
                },
            },
        },

        relationships: {
            HAS_CHUNK: {
                from: "Message",
                to: "Chunk",
                cardinality: "1..n",
                relProperties: { index: "number" },
            },
            SENT_BY: {
                from: "Message",
                to: "Person",
                cardinality: "0..1",
            },
            MENTIONS: {
                from: "Chunk",
                to: "Entity",
                cardinality: "0..n",
            },
            REFERENCES: {
                from: "Message",
                to: "Document",
                cardinality: "0..n",
            },
            REPLY_TO: {
                from: "Message",
                to: "Message",
                cardinality: "0..1",
            },
        },
    },
    async onDelete(ctx, data) {
        ctx.unlinkAll(data.id);
        ctx.deleteNode(data.id);
    },
});
