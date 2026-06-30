import { defineModule } from "../../../core/module";

// Shared generic entity node, lifted out of TextFile's local `Entity` so the
// same entity can be MENTIONED across docs, chat and tickets (cross-source
// resolution). Kept GENERIC on purpose: today it carries only id + name; the
// optimizer-controlled LLM-extraction lever later adds an `<entity_type>` label
// and typed Entity->Entity relations (the commented-out hints in
// textFile.module.ts). Until then this is the catch-all for gazetteer / regex
// tokens that are not resolved to a Person or Component.
export const EntityModule = defineModule({
    schema: {
        name: "EntityModule",
        root: "Entity",

        nodes: {
            Entity: {
                // Triplet extraction (LLM lever, added later):
                // labels: ["Entity", "<entity_type>"],
                labels: ["Entity"],
                properties: {
                    id: "string",
                    name: "string?",
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
