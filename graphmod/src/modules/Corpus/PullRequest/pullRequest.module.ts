import { defineModule } from "../../../core/module";
import { PersonModule } from "../../Person/person.module";
import { ComponentModule } from "../Component/component.module";
import { EntityModule } from "../Entity/entity.module";
import { ChunkModule } from "../Chunk/chunk.module";
import { DocModule } from "../Doc/doc.module";
import { TicketModule } from "../Ticket/ticket.module";

// GitHub pull-request source. Structured fields give author -> Person,
// reviewers -> Person, fixes -> Ticket (the PR that resolves a Jira ticket),
// component -> Component, and doc links -> Document. The PR body/description is
// the card chunk; MENTIONS picks up gazetteer/regex entities (e.g. PROJ-\d+).
export const PullRequestModule = defineModule({
    schema: {
        name: "PullRequestModule",
        root: "PullRequest",

        include: {
            Person: PersonModule,
            Component: ComponentModule,
            Entity: EntityModule,
            Chunk: ChunkModule,
            Document: DocModule,
            Ticket: TicketModule,
        },

        nodes: {
            PullRequest: {
                labels: ["PullRequest"],
                properties: {
                    id: "string",
                    number: "number",
                    title: "string",
                    repo: "string?",
                    state: "string?",
                    description: "string?",
                },
            },
        },

        relationships: {
            HAS_CHUNK: {
                from: "PullRequest",
                to: "Chunk",
                cardinality: "1..n",
                relProperties: { index: "number" },
            },
            AUTHORED_BY: {
                from: "PullRequest",
                to: "Person",
                cardinality: "0..1",
            },
            REVIEWED_BY: {
                from: "PullRequest",
                to: "Person",
                cardinality: "0..n",
            },
            FIXES: {
                from: "PullRequest",
                to: "Ticket",
                cardinality: "0..n",
            },
            ABOUT: {
                from: "PullRequest",
                to: "Component",
                cardinality: "0..n",
            },
            REFERENCES: {
                from: "PullRequest",
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
