import { defineModule } from "../../core/module";
import { PersonModule } from "../Person/person.module";




export const MicrosoftAccountModule = defineModule({
    schema: {
        name: "MicrosoftAccountModule",
        root: "MicrosoftAccount",

        include: {
            Person: PersonModule,
        },

        nodes: {
            MicrosoftAccount: {
                labels: ["MicrosoftAccount"],
                properties: {
                    id: "string",  
                    userPrincipalName: "string",
                    userType: "string?",
                    creationType: "string?",
                    email: "string?",
                    displayName: "string?",
                    accountEnabled: "boolean?",
                    createdAt: "datetime",
                    lastModifiedAt: "datetime",

                },
            },
        },

        relationships: {
            ACCOUNT_OF: {
                from: "MicrosoftAccount",
                to: "Person",
                cardinality: "0..1",
            }
        },
    },
    async onDelete(ctx, data) {
        ctx.unlinkAll(data.id)
        ctx.deleteNode(data.id)
    }
})
