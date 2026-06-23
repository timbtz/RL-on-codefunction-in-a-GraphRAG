// ============================================================
// video-module-seed.cypher — Sample data for VideoModule
// ============================================================
//
// Graph shape produced:
//
//   Person ──SPEAKS_IN──▶ Chunk:VideoChunk ──MENTIONS──▶ Entity
//                               ▲
//   Document:Video ──HAS_CHUNK──┘
//
// Ownership rules:
//   owned  (Document:Video, Chunk:VideoChunk) — deleted with the module
//   shared (Person, Entity)                   — only unlinked, never deleted
//
// Run in Neo4j Browser or cypher-shell:
//   cypher-shell -u neo4j -p password < video-module-seed.cypher

// ── 1. Shared nodes (MERGE — survive module delete) ─────────────────────────

MERGE (alice:Person  {id: 'person-alice'})
  ON CREATE SET alice.name  = 'Alice Müller',    alice.email = 'alice@example.com'
MERGE (bob:Person    {id: 'person-bob'})
  ON CREATE SET bob.name    = 'Bob Schmidt',     bob.email   = 'bob@example.com'

MERGE (eOkrs:Entity   {id: 'entity-okrs'})        ON CREATE SET eOkrs.name   = 'OKRs',           eOkrs.type   = 'Concept'
MERGE (eQ3:Entity     {id: 'entity-q3-2026'})     ON CREATE SET eQ3.name     = 'Q3 2026',        eQ3.type     = 'TimePeriod'
MERGE (eBerlin:Entity {id: 'entity-berlin'})      ON CREATE SET eBerlin.name = 'Berlin office',  eBerlin.type = 'Organization'
MERGE (eMl:Entity     {id: 'entity-ml-platform'}) ON CREATE SET eMl.name     = 'ML Platform',    eMl.type     = 'Product'
MERGE (eQ4:Entity     {id: 'entity-q4-budget'})   ON CREATE SET eQ4.name     = 'Q4 Budget',      eQ4.type     = 'Financial'

WITH alice, bob, eOkrs, eQ3, eBerlin, eMl, eQ4

// ── 2. Owned root — Document:Video ──────────────────────────────────────────
//    Schema properties: id (string), source (string?)
//    Extra runtime properties written by addRecording: title, durationSeconds

CREATE (doc:Document:Video {
  id:              'video-demo-001',
  source:          'https://teams.microsoft.com/rec/demo-001',
  title:           'Q3 Planning – Full Recording',
  durationSeconds: 3600,
  createdAt:       '2026-05-25T10:00:00Z',
  updatedAt:       '2026-05-25T10:00:00Z'
})

// ── 3. Owned chunks — Chunk:VideoChunk ──────────────────────────────────────
//    Schema properties: text (string)
//    Extra runtime properties: startTime, endTime, index

CREATE (c0:Chunk:VideoChunk {
  id:        'chunk-demo-001-0',
  text:      "Let's review the OKRs for next quarter.",
  startTime: 0,
  endTime:   12.5,
  index:     0,
  createdAt: '2026-05-25T10:00:00Z',
  updatedAt: '2026-05-25T10:00:00Z'
})
CREATE (c1:Chunk:VideoChunk {
  id:        'chunk-demo-001-1',
  text:      'The Berlin office wants to expand the ML platform.',
  startTime: 12.5,
  endTime:   24.0,
  index:     1,
  createdAt: '2026-05-25T10:00:00Z',
  updatedAt: '2026-05-25T10:00:00Z'
})
CREATE (c2:Chunk:VideoChunk {
  id:        'chunk-demo-001-2',
  text:      'We need to align on the Q4 budget before the board meeting.',
  startTime: 24.0,
  endTime:   38.5,
  index:     2,
  createdAt: '2026-05-25T10:00:00Z',
  updatedAt: '2026-05-25T10:00:00Z'
})

// ── 4. HAS_CHUNK edges  (Document:Video 1..n Chunk) ─────────────────────────

CREATE (doc)-[:HAS_CHUNK]->(c0)
CREATE (doc)-[:HAS_CHUNK]->(c1)
CREATE (doc)-[:HAS_CHUNK]->(c2)

// ── 5. SPEAKS_IN edges  (Person 0..n Chunk) ─────────────────────────────────
//    c2 has two speakers to exercise the multi-speaker case

CREATE (alice)-[:SPEAKS_IN]->(c0)
CREATE (bob)-[:SPEAKS_IN]->(c1)
CREATE (alice)-[:SPEAKS_IN]->(c2)
CREATE (bob)-[:SPEAKS_IN]->(c2)

// ── 6. MENTIONS edges   (Chunk 0..n Entity) ─────────────────────────────────

CREATE (c0)-[:MENTIONS]->(eOkrs)
CREATE (c0)-[:MENTIONS]->(eQ3)
CREATE (c1)-[:MENTIONS]->(eBerlin)
CREATE (c1)-[:MENTIONS]->(eMl)
CREATE (c2)-[:MENTIONS]->(eQ4)
