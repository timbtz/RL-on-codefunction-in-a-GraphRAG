import { defineModule } from "../../core/module";

// A thin, standalone shared node. It declares NO content-specific relations,
// so it never has to import (or depend on) any content module. Edges that
// concern a content type are declared *in that content module* instead — either
// as an inbound module ref (Option A) or by `include`-ing this module (Option B).
export const PersonModule = defineModule({
    schema: {
        name: "PersonModule",
        root: "Person",

        nodes: {
            Person: {
                labels: ["Person"],
                properties: {
                    id: "string",
                    name: "string",
                    info: "string?"
                },
            },
        },

        relationships: {},
    },
    async onDelete(ctx, data) {
        ctx.unlinkAll(data.id)
        ctx.deleteNode(data.id)
    }
})
