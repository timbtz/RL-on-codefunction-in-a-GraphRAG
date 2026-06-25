import { defineModule } from "../../core/module";
import { PersonModule } from "../Person/person.module";



export const GithubAccountModule = defineModule({
    schema: {
        name: "GithubAccountModule",
        root: "GithubAccount",

        include: {
            Person: PersonModule,
        },
        nodes: {
            GithubAccount: {
                labels: ["GithubAccount"],
                properties: {
                    id: "string",
                }
            },
        },
        relationships: {
            ACCOUNT_OF: {
                from: "GithubAccount",
                to: "Person",
                cardinality: "1",
            }
        },
    },
    async onDelete(ctx, data) {
        ctx.unlinkAll(data.id)
        ctx.deleteNode(data.id)
    }
})

