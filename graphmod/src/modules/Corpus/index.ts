// Barrel + explicit module list for the corpus graph. `export-schema.ts` and the
// ingestion layer import CORPUS_MODULES to union all labels / relationships into
// schema.json (the allowed-vocabulary the extractor + attribution probe rely on).
import type { AnyModule } from "../../core/types";
import { PersonModule } from "../Person/person.module";
import { ComponentModule } from "./Component/component.module";
import { EntityModule } from "./Entity/entity.module";
import { ChunkModule } from "./Chunk/chunk.module";
import { DocModule } from "./Doc/doc.module";
import { MessageModule } from "./Message/message.module";
import { TicketModule } from "./Ticket/ticket.module";

export {
    PersonModule,
    ComponentModule,
    EntityModule,
    ChunkModule,
    DocModule,
    MessageModule,
    TicketModule,
};

// Order matters only for readability; export-schema unions them.
export const CORPUS_MODULES: AnyModule[] = [
    PersonModule,
    ComponentModule,
    EntityModule,
    ChunkModule,
    DocModule,
    MessageModule,
    TicketModule,
];
