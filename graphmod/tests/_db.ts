// ============================================================
// _db.ts — Neo4j session lifecycle for the test suite
// ============================================================
//
// Tests use Node's built-in test runner (`node:test`), run through `tsx`.
// This helper owns the driver so a test file just does:
//
//   let session: GraphSession;
//   before(() => { session = openSession(); });
//   after(closeDriver);
//
// Connection comes from `.env` (loaded via `dotenv/config`):
//   NEO4J_URI, NEO4J_USER, NEO4J_PASS, NEO4J_DB
// Defaults: bolt://localhost:7687 / neo4j / password / neo4j
//
import { driver, auth, type Driver } from "neo4j-driver";
import type { GraphSession } from "../src/index";

let _driver: Driver | null = null;

export function openSession(): GraphSession {
  _driver = driver(
    process.env.NEO4J_URI ?? "bolt://localhost:7687",
    auth.basic(
      process.env.NEO4J_USER ?? "neo4j",
      process.env.NEO4J_PASS ?? "password",
    ),
  );
  return _driver.session({ database: process.env.NEO4J_DB ?? "neo4j" });
}

export async function closeDriver(): Promise<void> {
  await _driver?.close();
  _driver = null;
}
