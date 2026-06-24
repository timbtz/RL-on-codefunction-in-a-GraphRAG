import { VideoFileModule } from "../src/modules/Document/VideoFile/videoFile.module";
import { getSession } from "./session";


(async function () {

    const session = await getSession();

    // ---- Option A: inbound module ref (videoFile.module.ts) ----
    // SPEAKS_IN is declared in VideoFileModule as an inbound ref. The speakers
    // are reached inbound from a Chunk; each Person is PersonModule-hydrated.
    const doc = await VideoFileModule.fromLazyGraph(session, "doc-video-1");
    console.log(doc);

    const speakers = await doc.HAS_CHUNK.at(0)._reverse.SPEAKS_IN;
    console.log(speakers);                                   // Person[]
    console.log(await doc.HAS_CHUNK.at(0)._reverse.SPEAKS_IN.at(0)._rel.role);

    doc
        .HAS_CHUNK.at(0)
        ._reverse.SPEAKS_IN.at(0)

    // Outbound module ref on the same module: Chunk ──IS_SPEAKER──> Person.
    // Reached forward, directly on the node (no `_reverse`).
})()
