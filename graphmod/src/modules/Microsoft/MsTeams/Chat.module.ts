import { defineModule } from "../../../core/module";
import { TextFileModule } from "../../Document/TextFile/textFile.module";
import { VideoFileModule } from "../../Document/VideoFile/videoFile.module";
import { MicrosoftAccountModule } from "../MicrosoftAccount.module";



export const ChatModule = defineModule({
    schema: {
        name: "ChatModule",
        root: "Chat",

        include: {
            MicrosoftAccount: MicrosoftAccountModule,
            TextFile: TextFileModule,
            VideoFile: VideoFileModule
        },

        nodes: {
            Chat: {
                labels: ["Chat"],
                properties: {
                    id: "string",
                    chatId: "string",
                }
            },
            Message: {
                labels: ["Message"],
                properties: {
                    id: "string",
                    messageId: "string",
                }
            },
            Attachment: {
                labels: ["Attachment"],
                properties: {
                    id: "string",
                    name: "string",
                    contentType: "string",
                }
            }
        },
        relationships: {
            HAS_MESSAGE: {
                from: "Chat",
                to: "Message",
                cardinality: "0..n",
            },
            SENT: {
                from: "MicrosoftAccount",
                to: "Message",
                cardinality: "0..n",
            },
            MENTIONED_IN: {
                from: "MicrosoftAccount",
                to: "Message",
                cardinality: "0..n",
            },
            HAS_ATTACHMENT: {
                from: "Message",
                to: "Attachment",
                cardinality: "0..n",
            },
            PROCESSED_AS_TEXTFILE: {
                from: "Attachment",
                to: "TextFile",
                cardinality: "0..1",
            },
            PROCESSED_AS_VIDEOFILE: {
                from: "Attachment",
                to: "VideoFile",
                cardinality: "0..1",
            }
        }

    },
    async onDelete(ctx, data) {
        const messages = await data.HAS_MESSAGE
        for await (const message of messages) {
            const attachments = await message.HAS_ATTACHMENT
            for await (const attachment of attachments) {
                const textFile = await attachment.PROCESSED_AS_TEXTFILE
                if(textFile)
                    await ctx.deleteModule(TextFileModule, textFile.id)

                const videoFile = await attachment.PROCESSED_AS_VIDEOFILE
                if(videoFile)
                    await ctx.deleteModule(VideoFileModule, videoFile.id)

                ctx.unlinkAll(attachment.id)
                ctx.deleteNode(attachment.id)
            }

            ctx.unlinkAll(message.id)
            ctx.deleteNode(message.id)
        }

        ctx.unlinkAll(data.id)
        ctx.deleteNode(data.id)
    }
})

