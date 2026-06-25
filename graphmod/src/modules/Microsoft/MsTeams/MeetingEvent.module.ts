import { defineModule } from "../../../core/module";
import { VideoFileModule } from "../../Document/VideoFile/videoFile.module";
import { MicrosoftAccountModule } from "../MicrosoftAccount.module";


export const MeetingEventModule = defineModule({
    schema: {
        name: "MeetingEventModule",
        root: "MeetingEvent",

        include: {
            VideoFile: VideoFileModule,
            MicrosoftAccount: MicrosoftAccountModule,
        },

        nodes: {
            MeetingEvent: {
                labels: ["MeetingEvent"],
                properties: {
                    id: "string",
                    eventId: "string",
                    scheduledStartTime: "string",
                    scheduledEndTime: "string",
                    isCancelled: "boolean",
                },
            },
            Recording: {
                labels: ["Recording"],
                properties: {
                    id: "string",
                }
            }
            
        },

        relationships: {
            HAS_RECORDING: {
                from: "MeetingEvent",
                to: "Recording",
                cardinality: "0..n",
            },
            PROCESSED_AS_VIDEOFILE: {
                from: "Recording",
                to: "VideoFile",
                cardinality: "0..1",
            },
            PARTICIPATED_IN: {
                from: "MicrosoftAccount",
                to: "MeetingEvent",
                cardinality: "0..n",
            },
        }
    },
    async onDelete(ctx, data) {
        const recordings = await data.HAS_RECORDING
        for await (const recording of recordings) {
            const videoFile = await recording.PROCESSED_AS_VIDEOFILE
            if(videoFile)
                await ctx.deleteModule(VideoFileModule, videoFile.id)
            
            ctx.unlinkAll(recording.id)
            ctx.deleteNode(recording.id)
        }

        ctx.unlinkAll(data.id)
        ctx.deleteNode(data.id)


    }
})

