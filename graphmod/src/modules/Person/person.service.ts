// ============================================================
// person.service.ts — Domain service for the Person module
// ============================================================
import * as crypto from "node:crypto"; // Node 18: no global WebCrypto — needed for crypto.randomUUID()
//
// Graph shape:
//
//   (:Person { id, name, info? })
//
// A Person is a thin, shared node with no content-specific relations, so the
// service only deals with the root node itself.
//
import { type GraphSession, type GraphTransaction } from "../../index";
import { PersonModule } from "./person.module";

// ---- Inputs ----

export interface CreatePersonInput {
  name: string;
  info?: string;
}

/** Mutable properties of a Person. `id` is immutable, so it is excluded. */
export interface PersonProperties {
  name?: string;
  info?: string;
}

export class PersonService {
  /**
   * Create a Person node.
   *
   * Pass a `GraphSession` to run in its own transaction (standalone use).
   * Pass a `GraphTransaction` to participate in the caller's transaction —
   * useful when another service has already created nodes that this call
   * needs to commit atomically with.
   *
   * @returns the new Person node ID
   */
  static async create(
    target: GraphSession | GraphTransaction,
    input: CreatePersonInput,
  ): Promise<string> {
    const id = crypto.randomUUID();

    const work = async (tx: GraphTransaction) => {
      await tx.run(
        `CREATE (p:Person {
           id: $id,
           name: $name,
           info: $info
         })`,
        { id, name: input.name, info: input.info ?? null },
      );

      // Validate the freshly-written node inside the same tx. If a required
      // property is missing or mistyped, this throws `SchemaValidationError`
      // and rolls back every write above (including a parent tx's pre-writes
      // when sharing their transaction).
      await PersonModule.assertValid(tx, id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return id;
  }

  /**
   * Update mutable properties on an existing Person.
   *
   * Only the keys present in `properties` are written — undefined keys are
   * left untouched. The subgraph is re-validated in-tx after the update, so an
   * update that violates the schema rolls back.
   */
  static async set(
    target: GraphSession | GraphTransaction,
    id: string,
    properties: PersonProperties,
  ): Promise<void> {
    const work = async (tx: GraphTransaction) => {
      // `+=` merges only the supplied keys onto the existing node, leaving
      // the rest in place.
      await tx.run(
        `MATCH (p:Person { id: $id })
         SET p += $properties`,
        { id, properties },
      );

      await PersonModule.assertValid(tx, id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }
  }

  /**
   * Test helper: creates a Person with a deliberately missing required
   * property (`name`, which the schema marks as `string` — not optional).
   *
   * In-tx validation sees the missing property and throws
   * `SchemaValidationError`, rolling back every write in this call.
   */
  static async createWithMissingProperty(
    target: GraphSession | GraphTransaction,
  ): Promise<string> {
    const id = crypto.randomUUID();

    const work = async (tx: GraphTransaction) => {
      // The mistake: create a Person without the required `name`.
      await tx.run(
        `CREATE (p:Person {
           id: $id,
           info: $info
         })`,
        { id, info: "person created without the required name property" },
      );

      await PersonModule.assertValid(tx, id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return id;
  }
}
