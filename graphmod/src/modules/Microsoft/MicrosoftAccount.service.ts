// ============================================================
// MicrosoftAccount.service.ts — Domain service for the MicrosoftAccount module
// ============================================================
import * as crypto from "node:crypto"; // Node 18: no global WebCrypto — needed for crypto.randomUUID()
//
// Graph shape:
//
//   (:MicrosoftAccount { id, userPrincipalName, ... })-[:ACCOUNT_OF]->(:Person)
//
// ACCOUNT_OF is cardinality "1": every MicrosoftAccount belongs to exactly one
// Person. This service owns the edge that links the two.
//
import { type GraphSession, type GraphTransaction } from "../../index";
import { MicrosoftAccountModule } from "./MicrosoftAccount.module";

export class MicrosoftAccountService {

  static async upsertAccount(
    target: GraphSession | GraphTransaction,
    accountData: {
      id: string;
      userPrincipalName: string;
      displayName?: string;
      email?: string;
      userType?: string;
      creationType?: string;
      accountEnabled?: boolean;
    },
  ): Promise<string> {
    const work = async (tx: GraphTransaction) => {
      const props = Object.fromEntries(
        Object.entries(accountData).filter(([, v]) => v !== undefined),
      );

      await tx.run(
        `MERGE (a:MicrosoftAccount { id: $id })
         ON CREATE SET a.createdAt = datetime()
         SET a += $props, a.lastModifiedAt = datetime()`,
        { id: accountData.id, props },
      );

      await MicrosoftAccountModule.assertValid(tx, accountData.id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return accountData.id;
  }


  static async _createDummyAccount(
    target: GraphSession | GraphTransaction,
  ): Promise<string> {
    const id = crypto.randomUUID();
    const handle = id.slice(0, 8);
    const displayName = `Dummy User ${handle}`;
    const userPrincipalName = `dummy.${handle}@contoso.onmicrosoft.com`;
    const email = userPrincipalName;
    const isWorkSchool = Math.random() < 0.5;

    const work = async (tx: GraphTransaction) => {
      await tx.run(
        `WITH datetime() AS now
         CREATE (a:MicrosoftAccount {
           id: $id,
           userPrincipalName: $userPrincipalName,
           displayName: $displayName,
           email: $email,
           proxyAddresses: $proxyAddresses,
           accountType: $accountType,
           tenantId: $tenantId,
           userType: $userType,
           accountEnabled: $accountEnabled,
           createdAt: now,
           lastModifiedAt: now
         })`,
        {
          id,
          userPrincipalName,
          displayName,
          email,
          proxyAddresses: [`SMTP:${email}`, `smtp:${handle}@contoso.com`],
          accountType: isWorkSchool ? "workSchool" : "personal",
          tenantId: isWorkSchool ? crypto.randomUUID() : null,
          userType: Math.random() < 0.5 ? "Member" : "Guest",
          accountEnabled: true,
        },
      );

      await MicrosoftAccountModule.assertValid(tx, id);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }

    return id;
  }


  static async linkAccountToPerson(
    target: GraphSession | GraphTransaction,
    accountId: string,
    personId: string,
  ): Promise<void> {
    const work = async (tx: GraphTransaction) => {
      const found = await tx.run(
        `MATCH (a:MicrosoftAccount { id: $accountId })
         MATCH (p:Person { id: $personId })
         MERGE (a)-[:ACCOUNT_OF]->(p)
         RETURN a.id AS accountId, p.id AS personId`,
        { accountId, personId },
      );

      if (found.records.length === 0) {
        throw new Error(
          `Cannot link ACCOUNT_OF: MicrosoftAccount "${accountId}" or Person "${personId}" not found`,
        );
      }

      // Re-validate the account subgraph in-tx. A cardinality or schema
      // violation throws `SchemaValidationError` and rolls back the MERGE above.
      await MicrosoftAccountModule.assertValid(tx, accountId);
    };

    if ("executeWrite" in target) {
      await target.executeWrite(work);
    } else {
      await work(target);
    }
  }
}
