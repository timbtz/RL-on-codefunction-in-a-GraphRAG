// ============================================================
// MeetingEvent.service.ts — Domain service for the MeetingEvent module
// ============================================================
//
// Graph shape (MeetingEventModule):
//
//   (:MeetingEvent)-[:HAS_RECORDING]->(:Recording)              // HAS_RECORDING          "0..n"
//   (:Recording)-[:PROCESSED_AS_VIDEOFILE]->(:VideoFile)        // PROCESSED_AS_VIDEOFILE "0..1"
//   (:MicrosoftAccount)-[:PARTICIPATED_IN]->(:MeetingEvent)     // PARTICIPATED_IN        "0..n"
//
// A MeetingEvent is one concrete occurrence of a Meeting. This service owns the
// two edges that hang off it: recordings produced during the event, and the
// set of accounts that participated.
//
import { type GraphSession, type GraphTransaction } from "../../../index";
import { MeetingEventModule } from "./MeetingEvent.module";

export class MeetingEventService {
  /**
   * Add a Recording to an existing MeetingEvent via `HAS_RECORDING`.
   *
   * The `Recording` node carries no declared schema properties, but is given a
   * generated `id` so the returned handle can later be linked to a processed
   * `VideoFile` (`PROCESSED_AS_VIDEOFILE`) or matched again. The MeetingEvent
   * subgraph is validated in-tx after the write.
   *
   * Pass a `GraphSession` to run in its own transaction (standalone use).
   * Pass a `GraphTransaction` to participate in the caller's transaction.
   *
   * @param meetingEventId  `id` of the parent MeetingEvent
   * @returns the new Recording node ID
   */
  static async addRecording(
    target: GraphSession | GraphTransaction,
    meetingEventId: string,
  ): Promise<string> {
    const id = crypto.randomUUID();

    const work = async (tx: GraphTransaction) => {
      const result = await tx.run(
        `MATCH (e:MeetingEvent { id: $meetingEventId })
         CREATE (r:Recording { id: $id })
         MERGE (e)-[:HAS_RECORDING]->(r)
         RETURN r.id AS id`,
        { id, meetingEventId },
      );

      if (result.records.length === 0) {
        throw new Error(
          `Cannot add Recording: MeetingEvent "${meetingEventId}" not found`,
        );
      }

      await MeetingEventModule.assertValid(tx, meetingEventId);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return id;
  }

  /**
   * Replace the set of participants on a MeetingEvent.
   *
   * `PARTICIPATED_IN` is `0..n`, so this sets the full participant list: every
   * existing `(:MicrosoftAccount)-[:PARTICIPATED_IN]->(event)` edge is removed,
   * then one edge is `MERGE`d for each account in `accountIds`. Passing an empty
   * array clears all participants. Every account must already exist — a missing
   * one throws and nothing is changed. The subgraph is validated in-tx.
   *
   * @param meetingEventId  `id` of the MeetingEvent
   * @param accountIds      the complete set of participating MicrosoftAccount ids
   */
  static async updateParticipants(
    target: GraphSession | GraphTransaction,
    meetingEventId: string,
    accountIds: string[],
  ): Promise<void> {
    const work = async (tx: GraphTransaction) => {
      const eventCheck = await tx.run(
        `MATCH (e:MeetingEvent { id: $meetingEventId }) RETURN e.id AS id`,
        { meetingEventId },
      );
      if (eventCheck.records.length === 0) {
        throw new Error(
          `Cannot update participants: MeetingEvent "${meetingEventId}" not found`,
        );
      }

      // Verify every supplied account exists before mutating anything.
      if (accountIds.length > 0) {
        const matched = await tx.run(
          `UNWIND $accountIds AS aid
           MATCH (a:MicrosoftAccount { id: aid })
           RETURN collect(a.id) AS ids`,
          { accountIds },
        );
        const foundIds: string[] = matched.records[0]?.get("ids") ?? [];
        if (foundIds.length !== accountIds.length) {
          const missing = accountIds.filter((aid) => !foundIds.includes(aid));
          throw new Error(
            `Cannot update participants: MicrosoftAccount(s) not found: ${missing.join(", ")}`,
          );
        }
      }

      // Clear the existing participant set...
      await tx.run(
        `MATCH (:MicrosoftAccount)-[r:PARTICIPATED_IN]->(e:MeetingEvent { id: $meetingEventId })
         DELETE r`,
        { meetingEventId },
      );

      // ...then attach the new one (idempotent per account via MERGE).
      if (accountIds.length > 0) {
        await tx.run(
          `MATCH (e:MeetingEvent { id: $meetingEventId })
           UNWIND $accountIds AS aid
           MATCH (a:MicrosoftAccount { id: aid })
           MERGE (a)-[:PARTICIPATED_IN]->(e)`,
          { meetingEventId, accountIds },
        );
      }

      await MeetingEventModule.assertValid(tx, meetingEventId);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }
  }
}
