// ============================================================
// Meeting.service.ts — Domain service for the Meeting module
// ============================================================
import * as crypto from "node:crypto"; // Node 18: no global WebCrypto — needed for crypto.randomUUID()
//
// Graph shape (MeetingModule):
//
//   (:MicrosoftAccount)-[:ORGANIZED]->(:Meeting)      // ORGANIZED  "1"
//   (:Meeting)-[:HAS_EVENT]->(:MeetingEvent)          // HAS_EVENT  "0..n"
//   (:Meeting)-[:HAS_CHAT]->(:Chat)                   // HAS_CHAT   "0..1"
//
// A Meeting is the recurring/standalone calendar object; each concrete
// occurrence is a MeetingEvent hung off it via HAS_EVENT. Every Meeting is
// ORGANIZED by exactly one MicrosoftAccount, so `createMeeting` requires the
// organizer's account id and wires that edge atomically with the node.
//
import { type GraphSession, type GraphTransaction } from "../../../index";
import { MeetingModule } from "./Meeting.module";

// ---- Inputs ----

export interface CreateMeetingInput {
  /** The organizing MicrosoftAccount — the meeting's `ORGANIZED` source. */
  organizerAccountId: string;
  iCalUId: string;
  isRecurring: boolean;
  title: string;
  onlineMeetingId?: string;
  joinUrl?: string;
}

/** Mutable Meeting properties. `id` is immutable, so it is excluded. */
export interface MeetingProperties {
  iCalUId?: string;
  isRecurring?: boolean;
  title?: string;
  onlineMeetingId?: string;
  joinUrl?: string;
}

export interface AddMeetingEventInput {
  eventId: string;
  scheduledStartTime: string;
  scheduledEndTime: string;
  isCancelled: boolean;
}

export class MeetingService {
  /**
   * Create a Meeting and link it to its organizer via `ORGANIZED`.
   *
   * The organizer MicrosoftAccount must already exist — the node and edge are
   * created in a single statement driven by a `MATCH` on the organizer, so a
   * missing organizer creates nothing and throws. The subgraph is validated
   * in-tx afterwards.
   *
   * Pass a `GraphSession` to run in its own transaction (standalone use).
   * Pass a `GraphTransaction` to participate in the caller's transaction.
   *
   * @returns the new Meeting node ID
   */
  static async createMeeting(
    target: GraphSession | GraphTransaction,
    input: CreateMeetingInput,
  ): Promise<string> {
    const id = crypto.randomUUID();

    const work = async (tx: GraphTransaction) => {
      const result = await tx.run(
        `MATCH (organizer:MicrosoftAccount { id: $organizerAccountId })
         CREATE (m:Meeting {
           id: $id,
           iCalUId: $iCalUId,
           isRecurring: $isRecurring,
           title: $title,
           onlineMeetingId: $onlineMeetingId,
           joinUrl: $joinUrl
         })
         MERGE (organizer)-[:ORGANIZED]->(m)
         RETURN m.id AS id`,
        {
          id,
          organizerAccountId: input.organizerAccountId,
          iCalUId: input.iCalUId,
          isRecurring: input.isRecurring,
          title: input.title,
          onlineMeetingId: input.onlineMeetingId ?? null,
          joinUrl: input.joinUrl ?? null,
        },
      );

      if (result.records.length === 0) {
        throw new Error(
          `Cannot create Meeting: organizer MicrosoftAccount "${input.organizerAccountId}" not found`,
        );
      }

      // Validate the freshly-written subgraph inside the same tx. A missing
      // required property (or an incomplete organizer account) throws
      // `SchemaValidationError` and rolls back every write above.
      await MeetingModule.assertValid(tx, id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return id;
  }

  /**
   * Update mutable properties on an existing Meeting.
   *
   * Only the keys present in `properties` are written — undefined keys are
   * left untouched (`SET m += $properties`). The subgraph is re-validated
   * in-tx, so an update that violates the schema rolls back.
   */
  static async updateMeeting(
    target: GraphSession | GraphTransaction,
    id: string,
    properties: MeetingProperties,
  ): Promise<void> {
    const work = async (tx: GraphTransaction) => {
      const result = await tx.run(
        `MATCH (m:Meeting { id: $id })
         SET m += $properties
         RETURN m.id AS id`,
        { id, properties },
      );

      if (result.records.length === 0) {
        throw new Error(`Cannot update Meeting: "${id}" not found`);
      }

      await MeetingModule.assertValid(tx, id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }
  }

  /**
   * Add a MeetingEvent occurrence to an existing Meeting via `HAS_EVENT`.
   *
   * Creates the MeetingEvent node and the `(:Meeting)-[:HAS_EVENT]->(:MeetingEvent)`
   * edge, then validates the Meeting subgraph in-tx — because MeetingModule
   * includes the MeetingEvent node, the new event's required properties are
   * checked here, and a violation rolls back both writes.
   *
   * @param meetingId  `id` of the parent Meeting
   * @returns the new MeetingEvent node ID
   */
  static async addMeetingEvent(
    target: GraphSession | GraphTransaction,
    meetingId: string,
    input: AddMeetingEventInput,
  ): Promise<string> {
    const id = crypto.randomUUID();

    const work = async (tx: GraphTransaction) => {
      const result = await tx.run(
        `MATCH (m:Meeting { id: $meetingId })
         CREATE (e:MeetingEvent {
           id: $id,
           eventId: $eventId,
           scheduledStartTime: $scheduledStartTime,
           scheduledEndTime: $scheduledEndTime,
           isCancelled: $isCancelled
         })
         MERGE (m)-[:HAS_EVENT]->(e)
         RETURN e.id AS id`,
        {
          id,
          meetingId,
          eventId: input.eventId,
          scheduledStartTime: input.scheduledStartTime,
          scheduledEndTime: input.scheduledEndTime,
          isCancelled: input.isCancelled,
        },
      );

      if (result.records.length === 0) {
        throw new Error(
          `Cannot add MeetingEvent: Meeting "${meetingId}" not found`,
        );
      }

      await MeetingModule.assertValid(tx, meetingId);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return id;
  }
}
