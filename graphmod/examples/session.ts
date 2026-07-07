
import {driver, auth} from "neo4j-driver"; // Only needed if you implement getSession() with neo4j-driver
import type { GraphSession } from "../src/index";



export async function getSession(): Promise<GraphSession> {
  const neo4jDriver = driver(
    process.env.NEO4J_URI ?? "bolt://localhost:7687",
    auth.basic(
      process.env.NEO4J_USER ?? "neo4j",
      process.env.NEO4J_PASS ?? "password",
    ),
  );
  const session = neo4jDriver.session({
    database: process.env.NEO4J_DB ?? "neo4j",
  });
  // Close the DRIVER (its connection pool keeps the node event loop alive)
  // when the session is closed, so CLI scripts like ingest.ts actually exit.
  const closeSession = session.close.bind(session);
  (session as { close: () => Promise<void> }).close = async () => {
    await closeSession();
    await neo4jDriver.close();
  };
  return session;
}