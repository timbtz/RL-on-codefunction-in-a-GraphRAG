import { defineModule } from "../../../core/module";
import { PersonModule } from "../../Person/person.module";
import { ComponentModule } from "../Component/component.module";
import { EntityModule } from "../Entity/entity.module";
import { ChunkModule } from "../Chunk/chunk.module";
import { DocModule } from "../Doc/doc.module";

// Jira ticket source. The richest structured source: assignee/reporter -> Person,
// component -> Component, blocks -> Ticket, and doc links -> Document all come
// for free from fields. MENTIONS over the ticket description chunk picks up
// gazetteer/regex entities.
export const TicketModule = defineModule({
    schema: {
        name: "TicketModule",
        root: "Ticket",

        include: {
            Person: PersonModule,
            Component: ComponentModule,
            Entity: EntityModule,
            Chunk: ChunkModule,
            Document: DocModule,
        },

        nodes: {
            Ticket: {
                labels: ["Ticket"],
                properties: {
                    id: "string",
                    key: "string",
                    title: "string",
                    description: "string?",
                    status: "string?",
                },
            },
        },

        relationships: {
            HAS_CHUNK: {
                from: "Ticket",
                to: "Chunk",
                cardinality: "1..n",
                relProperties: { index: "number" },
            },
            ASSIGNED: {
                from: "Ticket",
                to: "Person",
                cardinality: "0..1",
            },
            REPORTED: {
                from: "Ticket",
                to: "Person",
                cardinality: "0..1",
            },
            ABOUT: {
                from: "Ticket",
                to: "Component",
                cardinality: "0..n",
            },
            BLOCKS: {
                from: "Ticket",
                to: "Ticket",
                cardinality: "0..n",
            },
            REFERENCES: {
                from: "Ticket",
                to: "Document",
                cardinality: "0..n",
            },
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
