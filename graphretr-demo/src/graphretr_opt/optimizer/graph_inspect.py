"""graph_inspect.py -- a read-only inspection gate over the candidate's built
graph, plus the per-miss attribution helper.

This mirrors the STaRK Cypher firewall (`starksearch/src/stark_harness/qa_runner.py`:
`_assert_safe` + `ReadOnlyGraphClient`): every inspection query is asserted
read-only, force-bounded with a `LIMIT`, and truncated, so the SHARED Neo4j the
optimizer runs against can never be written to or pegged by an inspection probe.

Two consumers:
  * the reward adapter (`reward/ingest_search.py`) -- attributes each MISSED MCQ
    to NOT_INGESTED / ORPHANED / UNREACHABLE / RANKING by probing the built
    graph with the gold node id (the gold id is the ONLY gold signal that
    crosses in -- never the MCQ options or the answer index);
  * future ad-hoc graph debugging (schema_summary / count_by_label).

The graph object is injected (a `langchain_neo4j.Neo4jGraph`-like object exposing
`.query(cypher, params) -> list[dict]`), already scoped to the candidate's graph
database. We never build a driver here, so this module imports WITHOUT a live DB
(the connection is deferred to the injected graph). All Cypher is parameterized
and bounded; variable-length patterns clamp their hop count.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Firewall (mirror of starksearch qa_runner._assert_safe)                      #
# --------------------------------------------------------------------------- #
# Strip comments before scanning so a write keyword hidden in a comment cannot
# sneak a write past the regex (and a read primitive in a comment cannot trip a
# false positive).
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//[^\n]*")
# Write clauses + write procedures. Word-boundary so RETURN / queryNodes /
# CONTAINS (which embed none of these as whole words) pass cleanly.
_WRITE_RE = re.compile(r"\b(create|merge|set|delete|remove|drop|detach)\b")
_WRITE_PROC_RE = re.compile(
    r"call\s+(db\.create|apoc\.\w*(create|merge|set|delete|add|remove|update))")
# Does the query already carry a top-level LIMIT? (cheap heuristic: the word
# `limit` appears at all -- we only ever APPEND a LIMIT, never weaken one.)
_LIMIT_RE = re.compile(r"\blimit\b")

DEFAULT_MAX_ROWS = 50
DEFAULT_NEIGHBOR_DEPTH_CEILING = 3   # hard clamp on any variable-length probe walk


def assert_read_only(cypher: str) -> str:
    """THE FIREWALL. Reject a Cypher string that would write to (or via a write
    proc mutate) the shared graph. Raises ``ValueError`` on a violation; returns
    the original string otherwise.

    Mirrors ``starksearch...qa_runner._assert_safe``: comments are stripped and
    the body is lowercased before the write-clause / write-proc scan, so neither
    a commented-out write nor a read primitive inside a comment can fool it.
    """
    raw = str(cypher)
    q = _COMMENT_LINE.sub(" ", _COMMENT_BLOCK.sub(" ", raw)).lower()
    if _WRITE_RE.search(q) or _WRITE_PROC_RE.search(q):
        raise ValueError(f"unsafe cypher (write clause/proc): {raw[:200]}")
    return raw


def _slug_depth(depth: int, ceiling: int = DEFAULT_NEIGHBOR_DEPTH_CEILING) -> int:
    """Clamp a requested variable-length hop count into ``[1, ceiling]``. Cypher
    does not allow the range bound to be a parameter, so it is formatted into the
    pattern -- this keeps it ALWAYS bounded (never an open ``*``)."""
    try:
        d = int(depth)
    except (TypeError, ValueError):
        d = 1
    return max(1, min(d, max(1, int(ceiling))))


class GraphInspector:
    """Read-only, bounded inspector over the candidate's built graph.

    Args:
        graph: an injected ``Neo4jGraph``-like object exposing
            ``.query(cypher: str, params: dict) -> list[dict]``. It must already
            be scoped to the candidate's graph database (the adapter constructs it
            with ``database=graph_<hash>``). We never build a driver here.
        db: the graph database name this inspector is scoped to (informational --
            used only in attribution messages; the scoping itself lives in the
            injected graph). May be ``None``.
        max_rows: hard cap on rows returned by any ``inspect`` call (also the
            LIMIT appended when a query carries none).
    """

    def __init__(self, graph: Any, db: Optional[str] = None,
                 max_rows: int = DEFAULT_MAX_ROWS) -> None:
        self._graph = graph
        self._db = db
        self._max_rows = max(1, int(max_rows))

    # ------------------------------------------------------------------ #
    # The one chokepoint every inspection query passes through            #
    # ------------------------------------------------------------------ #
    def inspect(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        """Run ONE read-only, bounded Cypher query. -> list[dict] (rows),
        truncated to ``max_rows``.

        Firewall -> force a LIMIT (append ``LIMIT <max_rows>`` when the query
        carries none) -> run via the injected graph -> truncate. A write or write
        proc raises ``ValueError`` BEFORE the query reaches the graph.
        """
        safe = assert_read_only(cypher)
        if not _LIMIT_RE.search(safe.lower()):
            safe = f"{safe.rstrip().rstrip(';')} LIMIT {self._max_rows}"
        params = params if isinstance(params, dict) else {}
        rows = self._graph.query(safe, params)
        rows = list(rows) if rows is not None else []
        # rows are already dicts from Neo4jGraph; truncate defensively.
        return [dict(r) for r in rows[: self._max_rows]]

    @property
    def db(self) -> Optional[str]:
        return self._db

    # ------------------------------------------------------------------ #
    # Bounded helpers                                                     #
    # ------------------------------------------------------------------ #
    def find_node(self, node_id: str) -> Optional[dict]:
        """The node with ``id == node_id`` (canonical id, e.g. ``doc:auth``), or
        ``None``. Parameterized + LIMIT 1."""
        rows = self.inspect(
            "MATCH (n {id: $id}) RETURN n.id AS id, labels(n) AS labels LIMIT 1",
            {"id": node_id})
        return rows[0] if rows else None

    def neighbors(self, node_id: str, depth: int = 1) -> list[dict]:
        """Nodes reachable from ``node_id`` within a BOUNDED variable-length walk
        (depth clamped into ``[1, ceiling]``). -> list of ``{id, labels}`` dicts.

        The hop count is formatted into the pattern (Cypher forbids a parameter
        there) AFTER clamping, so the walk can never become an unbounded ``*``.
        Everything else stays parameterized and the inherited ``LIMIT`` bounds the
        result set.
        """
        d = _slug_depth(depth)
        cypher = (
            "MATCH (n {id: $id})-[*1.." f"{d}" "]-(m) "
            "RETURN DISTINCT m.id AS id, labels(m) AS labels")
        return self.inspect(cypher, {"id": node_id})

    def count_by_label(self) -> dict:
        """``{label: count}`` over the graph (one bounded row per label). Uses
        ``db.labels()`` then counts each -- both read-only."""
        labels = self.inspect("CALL db.labels() YIELD label RETURN label")
        out: dict = {}
        for row in labels:
            lbl = row.get("label")
            if not lbl:
                continue
            # backtick-escape the label (it comes from db.labels(), but stay safe)
            safe_lbl = "`" + str(lbl).replace("`", "") + "`"
            cnt = self.inspect(
                f"MATCH (n:{safe_lbl}) RETURN count(n) AS c")
            out[lbl] = int(cnt[0]["c"]) if cnt else 0
        return out

    def schema_summary(self) -> dict:
        """``{labels: [...], relationships: [...]}`` from ``db.labels()`` /
        ``db.relationshipTypes()`` -- a read-only snapshot for debugging / for
        constraining a downstream extractor."""
        labels = [r.get("label") for r in
                  self.inspect("CALL db.labels() YIELD label RETURN label")]
        rels = [r.get("relationshipType") for r in self.inspect(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType")]
        return {"labels": [l for l in labels if l],
                "relationships": [r for r in rels if r]}

    # ------------------------------------------------------------------ #
    # Attribution helper (the loop's INGESTION-vs-SEARCH signal)          #
    # ------------------------------------------------------------------ #
    def attribute_miss(self, gold_id: str,
                       seed_ids: Optional[list] = None,
                       retrieved_ids: Optional[list] = None) -> str:
        """Classify WHY the search missed ``gold_id`` for one question, by probing
        the built graph read-only. -> one of (with a short reason):

          * ``NOT_INGESTED`` -- the gold node id is absent from the graph. The fix
            is on the INGESTION side (add an extractor / edge / chunk card).
          * ``ORPHANED``     -- the gold node is present but has no path from any
            fulltext seed (here: no neighbours within the bounded walk, or -- when
            ``seed_ids`` is supplied -- no bounded path from any seed). INGESTION
            fix (wire it into the graph / give it a fulltext-indexed chunk).
          * ``RANKING``      -- the gold node WAS surfaced by search
            (``gold_id in retrieved_ids``) but ranked out of the returned context.
            SEARCH fix (scoring / context budget).
          * ``UNREACHABLE``  -- the gold node is present and connected in the graph
            but search did not reach it. SEARCH fix (depth / seed / neighbour
            limits).

        NEVER receives the MCQ options or the answer index -- only the gold NODE
        id (the id that proves the answer) plus the search's own seed/retrieved
        ids. The result string is what ``fast_loop._failure_record`` reads off
        ``row["graph_attribution"]`` (it prefers this over the recall-based split).
        """
        node = self.find_node(gold_id)
        if not node:
            return f"NOT_INGESTED (gold node {gold_id!r} absent from graph)"

        retrieved = {str(x) for x in (retrieved_ids or [])}
        if gold_id in retrieved:
            return (f"RANKING (gold {gold_id!r} retrieved by search but ranked "
                    "out of the returned context; fix scoring/context budget)")

        # ORPHANED: no path from any fulltext seed. With seeds we test a bounded
        # path from any seed; without them we fall back to "has any neighbour at
        # all" (a degree-0 node can never be reached from a seed).
        if seed_ids:
            if not self._has_path_from_seeds(gold_id, list(seed_ids)):
                return (f"ORPHANED (gold {gold_id!r} present but no bounded path "
                        "from any fulltext seed; fix ingestion edges)")
        else:
            if not self.neighbors(gold_id, depth=1):
                return (f"ORPHANED (gold {gold_id!r} present but has no neighbours "
                        "/ no path from any seed; fix ingestion edges)")

        return (f"UNREACHABLE (gold {gold_id!r} present and connected but search "
                "did not reach it; widen depth/seed/neighbour limits)")

    def _has_path_from_seeds(self, gold_id: str, seed_ids: list,
                             depth: int = DEFAULT_NEIGHBOR_DEPTH_CEILING) -> bool:
        """True iff some seed node has a BOUNDED path (<= depth hops) to the gold
        node. Bounded + parameterized seed list; the depth is clamped and
        formatted into the pattern (Cypher forbids a parameter there)."""
        if not seed_ids:
            return False
        d = _slug_depth(depth)
        cypher = (
            "MATCH (s) WHERE s.id IN $seeds "
            "MATCH (g {id: $gold}) "
            "MATCH p = (s)-[*1.." f"{d}" "]-(g) "
            "RETURN count(p) AS c")
        rows = self.inspect(cypher, {"seeds": list(seed_ids), "gold": gold_id})
        return bool(rows) and int(rows[0].get("c", 0) or 0) > 0
