import { defineModule } from "../../../core/module";
import { MicrosoftAccountModule } from "../MicrosoftAccount.module";
import { ChatModule } from "./Chat.module";
import { MeetingEventModule } from "./MeetingEvent.module";



export const MeetingModule = defineModule({
    schema: {
        name: "MeetingModule",
        root: "Meeting",

        include: {
            MicrosoftAccount: MicrosoftAccountModule,
            MeetingEvent: MeetingEventModule,
            Chat: ChatModule,
        },

        nodes: {
            Meeting: {
                labels: ["Meeting"],
                properties: {
                    id: "string",
                    iCalUId: "string",
                    isRecurring: "boolean",
                    title: "string",
                    joinUrl: "string?",
                    createdAt: "datetime",
                    lastModifiedAt: "datetime",
                }
            },
        },

        relationships: {
            ORGANIZED: {
                from: "MicrosoftAccount",
                to: "Meeting",
                cardinality: "1",
            },
            HAS_EVENT: {
                from: "Meeting",
                to: "MeetingEvent",
                cardinality: "0..n",
            },
            HAS_CHAT: {
                from: "Meeting",
                to: "Chat",
                cardinality: "0..1",
            }
        }
    },
    async onDelete(ctx, data) {
        const chat = await data.HAS_CHAT
        
        if(chat) 
            await ctx.deleteModule(ChatModule, chat.id)

        const events = await data.HAS_EVENT
        for await (const event of events)
            await ctx.deleteModule(MeetingEventModule, event.id)

        ctx.unlinkAll(data.id)
        ctx.deleteNode(data.id)
    }
})

