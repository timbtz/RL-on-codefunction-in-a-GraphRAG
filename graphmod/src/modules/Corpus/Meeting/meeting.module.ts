import { defineModule } from "../../../core/module";
import { PersonModule } from "../../Person/person.module";
import { ComponentModule } from "../Component/component.module";
import { EntityModule } from "../Entity/entity.module";
import { ChunkModule } from "../Chunk/chunk.module";
import { DocModule } from "../Doc/doc.module";
import { TicketModule } from "../Ticket/ticket.module";

// Teams meeting / daily standup source. Each meeting is a transcript: the card
// chunk names the attendees + the project ticket(s) discussed, and the body is
// split into per-utterance chunks. The SOLUTION to a problem is usually named in
// the transcript PROSE (a `Chunk`), not in a structured field — that is the gap
// the LLM-extraction lever can close (e.g. lift a (ticket)-[:RESOLVED_BY]->...).
export const MeetingModule = defineModule({
    schema: {
        name: "MeetingModule",
        root: "Meeting",

        include: {
            Person: PersonModule,
            Component: ComponentModule,
            Entity: EntityModule,
            Chunk: ChunkModule,
            Document: DocModule,
            Ticket: TicketModule,
        },

        nodes: {
            Meeting: {
                labels: ["Meeting"],
                properties: {
                    id: "string",
                    title: "string",
                    date: "string?",
                    project: "string?",
                },
            },
        },

        relationships: {
            HAS_CHUNK: {
                from: "Meeting",
                to: "Chunk",
                cardinality: "1..n",
                relProperties: { index: "number" },
            },
            ATTENDED: {
                from: "Meeting",
                to: "Person",
                cardinality: "0..n",
            },
            ABOUT: {
                from: "Meeting",
                to: "Component",
                cardinality: "0..n",
            },
            DISCUSSES: {
                from: "Meeting",
                to: "Ticket",
                cardinality: "0..n",
            },
            REFERENCES: {
                from: "Meeting",
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
