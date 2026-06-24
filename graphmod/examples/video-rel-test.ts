import { VideoFileModule } from "../src/modules/Document/VideoFile/videoFile.module";
import { PersonModule } from "../src/modules/Person/person.module";
import { getSession } from "./session";


(async function(){

    const session = await getSession();

    
    const videoFile = await VideoFileModule.fromLazyGraph(
        session,
        ""
    )

    videoFile.id;

    await videoFile
        .HAS_CHUNK.at(0)
        ._rel.index;



})()





