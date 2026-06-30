import { defineModule } from "../../../core/module";
import { PersonModule } from "../../Person/person.module";
import { ComponentModule } from "../Component/component.module";
import { EntityModule } from "../Entity/entity.module";
import { ChunkModule } from "../Chunk/chunk.module";

// Markdown document source. Mirrors TextFileModule (Document -> HAS_CHUNK ->
// Chunk -> MENTIONS -> Entity) and adds the cross-source edges the corpus needs:
// AUTHORED_BY (-> Person, the "owner" path) and ABOUT (-> Component), plus
// REFERENCES between documents (wikilinks / URLs).
export const DocModule = defineModule({
    schema: {
        name: "DocModule",
        root: "Document",

        include: {
            Person: PersonModule,
            Component: ComponentModule,
            Entity: EntityModule,
            Chunk: ChunkModule,
        },

        nodes: {
            Document: {
                labels: ["Document"],
                properties: {
                    id: "string",
                    title: "string",
                    path: "string?",
                },
            },
        },

        relationships: {
            HAS_CHUNK: {
                from: "Document",
                to: "Chunk",
                cardinality: "1..n",
                relProperties: { index: "number" },
            },
            MENTIONS: {
                from: "Chunk",
                to: "Entity",
                cardinality: "0..n",
            },
            AUTHORED_BY: {
                from: "Document",
                to: "Person",
                cardinality: "0..1",
            },
            ABOUT: {
                from: "Document",
                to: "Component",
                cardinality: "0..n",
            },
            REFERENCES: {
                from: "Document",
                to: "Document",
                cardinality: "0..n",
            },
        },
    },
    async onDelete(ctx, data) {
        // Eval rebuilds the graph per ingest-hash, so a shallow detach suffices.
        ctx.unlinkAll(data.id);
        ctx.deleteNode(data.id);
    },
});
