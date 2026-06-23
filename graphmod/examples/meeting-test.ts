import { VideoFileModule } from "../src/modules/Document/VideoFile/videoFile.module";
import { MicrosoftAccountService } from "../src/modules/Microsoft/MicrosoftAccount.service";
import { MeetingModule } from "../src/modules/Microsoft/MsTeams/Meeting.module";
import { MeetingService } from "../src/modules/Microsoft/MsTeams/Meeting.service";
import { MeetingEventModule } from "../src/modules/Microsoft/MsTeams/MeetingEvent.module";
import { MeetingEventService } from "../src/modules/Microsoft/MsTeams/MeetingEvent.service";
import { getSession } from "./session";


(async function () {

    const session = await getSession();

    // ---- Option A: inbound module ref (videoFile.module.ts) ----
    // SPEAKS_IN is declared in VideoFileModule as an inbound ref. The speakers
    // are reached inbound from a Chunk; each Person is PersonModule-hydrated.

    const acc = await MicrosoftAccountService._createDummyAccount(session)


    const meetingId = await MeetingService.createMeeting(session, {
        title: "GraphMod Sync",
        iCalUId: "1",
        isRecurring: false,
        onlineMeetingId: "1",
        joinUrl: "https://teams.microsoft.com/l/",
        organizerAccountId: acc,
    })

    


    let meeting = await MeetingModule.fromLazyGraph(session, meetingId);


    
        

    // meeting._reverse.ORGANIZED.at(0).ORGANIZED


    
    console.log("Events (before add):", await meeting.HAS_EVENT)


    const mEvent = await MeetingService.addMeetingEvent(session, meetingId, {
        eventId: "event-1",
        scheduledStartTime: new Date().toISOString(),
        scheduledEndTime: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        isCancelled: false,
    })

    meeting = await MeetingModule.fromLazyGraph(session, meetingId);

    console.log("Events (after add): ", await meeting.HAS_EVENT)



    await MeetingEventService.addRecording(session, mEvent)
    await MeetingEventService.addRecording(session, mEvent)
    await MeetingEventService.addRecording(session, mEvent)


    const events = await meeting
        .HAS_EVENT


    const organizer = await meeting
        ._reverse.ORGANIZED.at(0)

    console.log("Organizer account: ", organizer)
    meeting = await MeetingModule.fromLazyGraph(session, meetingId);

    const event = MeetingEventModule.fromLazyGraph(
        session,
        await meeting.HAS_EVENT.at(0).id
    );

    ;

    const recordings = await event.HAS_RECORDING;

    console.log("Recordings for event:", recordings)



        
        
})()
