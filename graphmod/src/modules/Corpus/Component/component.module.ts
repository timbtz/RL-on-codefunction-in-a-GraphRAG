import { defineModule } from "../../../core/module";

// Shared, standalone entity node (mirrors PersonModule). A `Component` is a
// resolved, controlled-vocabulary entity (e.g. a service/subsystem like
// "auth-redesign") that documents, chat messages and tickets can all point at.
// Like PersonModule it declares NO content-specific relations, so the content
// modules either `include` it or reference it from their own relationships.
export const ComponentModule = defineModule({
    schema: {
        name: "ComponentModule",
        root: "Component",

        nodes: {
            Component: {
                labels: ["Component"],
                properties: {
                    id: "string",
                    name: "string",
                    info: "string?",
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
