"""Editable seed search target — zero-LLM, rule-based fulltext seed + bounded traversal.

This module is the SEARCH half of the co-optimize loop (see
``Orchestration/Plans/cooptimize-CONTRACT.md``). It mirrors the production
``AgenticGraphTraversalSearchService`` (fulltext seed -> bounded graph walk) but
strips out every LLM call so the seed pipeline costs ``usd=0`` / ``tokens=0``.

It is the file the optimizer is allowed to edit, so:
  * it is a single, readable file,
  * the tunables live up top as clearly-commented constants / cfg keys,
  * there is exactly ONE disabled hook where an LLM keyword/expansion step
    could later be wired in (and where it would have to meter usd/tokens).

Contract anchors obeyed verbatim:
  * fulltext index name ............ ``corpus_ft``
  * seed nodes ..................... ``Chunk`` nodes (``node.id`` is the chunk id)
  * owner edge .................... ``HAS_CHUNK`` (owner -> chunk; we walk it backwards)
  * owner labels .................. ``Document`` / ``Message`` / ``Ticket``
  * corpus edges (bounded walk) ... AUTHORED_BY, ABOUT, REFERENCES, SENT_BY,
                                    ASSIGNED, REPORTED, BLOCKS, REPLY_TO, MENTIONS
  * citation token ................ ``[doc:<id>]`` / ``[msg:<id>]`` / ``[ticket:<id>]``

The injected ``graph`` object must expose ``.query(cypher: str, params: dict)
-> list[dict]`` exactly like ``langchain_neo4j.Neo4jGraph`` — we never build a
driver ourselves (the reward adapter injects a graph scoped to the candidate's
``graph_<hash>`` database).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# OPTIMIZER-EDITABLE TUNABLES                                                  #
# --------------------------------------------------------------------------- #
# These four are the headline levers the optimizer mutates (read from ``cfg``
# with these defaults). Keep them obvious and bounded.
DEFAULT_SEED_LIMIT: int = 20          # # of fulltext seed chunks kept (LIMIT on seed query)
DEFAULT_MAX_DEPTH: int = 2            # ALWAYS-bounded variable-length hop count for the walk
DEFAULT_NEIGHBOR_LIMIT: int = 25      # max # of owner nodes collected by the bounded walk
DEFAULT_MAX_CONTEXT_CHARS: int = 20_000  # hard cap on the returned context string

# Secondary safety bounds (kept conservative; also editable). Every Cypher query
# below carries a forced LIMIT so a runaway graph cannot blow up the candidate.
DEFAULT_MAX_CHUNKS_TOTAL: int = 200   # global LIMIT on chunk rows gathered for context
DEFAULT_DEPTH_CEILING: int = 4        # max_depth is clamped to [1, this] — keeps the walk bounded

# Rule-based keyword extraction config.
MIN_TOKEN_LEN: int = 3                # drop tokens shorter than this (after typed tokens are preserved)

# Typed tokens we ALWAYS preserve verbatim from the question (case-sensitive),
# even if they would otherwise be filtered (e.g. a ticket key or a chat handle):
#   * ``PROJ-\d+`` style ticket keys (any 2+ uppercase letters + '-' + digits)
#   * ``@handle`` style mentions
_TYPED_TOKEN_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"[A-Za-z]{2,}-\d+"),   # ticket keys, e.g. PROJ-12, ENG-7
    re.compile(r"@\w+"),                # chat handles, e.g. @alice
)

# Small, generic English stopword set — intentionally short so we do not strip
# domain terms. The optimizer may extend this.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
        "have", "has", "had", "not", "but", "you", "your", "all", "any", "can",
        "did", "does", "what", "when", "where", "which", "who", "whom", "why",
        "how", "into", "out", "about", "over", "than", "then", "they", "them",
        "their", "there", "here", "been", "being", "would", "could", "should",
        "will", "shall", "may", "might", "must", "its", "our", "his", "her",
        "she", "him", "had", "get", "got", "use", "used", "via", "per",
    }
)

# Corpus relationship types the bounded walk is allowed to traverse (undirected).
# Used verbatim in the variable-length Cypher pattern (see CONTRACT "Edges").
_CORPUS_EDGE_TYPES: tuple[str, ...] = (
    "AUTHORED_BY", "ABOUT", "REFERENCES", "SENT_BY", "ASSIGNED",
    "REPORTED", "BLOCKS", "REPLY_TO", "MENTIONS",
)

# Owner labels that bear answer-text + emit a citation token.
_OWNER_LABELS: tuple[str, ...] = ("Document", "Message", "Ticket")

# Label -> citation prefix. The canonical node ids already carry this prefix
# (e.g. ``doc:intro``), so the emitted token is ``[doc:intro]`` etc. We still
# key off the label so the prefix is authoritative per owner TYPE.
_CITATION_PREFIX_BY_LABEL: dict[str, str] = {
    "Document": "doc",
    "Message": "msg",
    "Ticket": "ticket",
}

# Lucene query-syntax special characters that must be escaped inside terms so a
# typed token like ``PROJ-12`` is matched literally (a bare ``-`` is the Lucene
# NOT operator).
_LUCENE_SPECIAL: re.Pattern = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


# --------------------------------------------------------------------------- #
# Result container (entrypoint contract for the reward adapter)               #
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    """Structured search output consumed by the reward adapter.

    ``usd`` / ``tokens`` are ZERO for the rule-based seed; they only become
    non-zero if the (currently disabled) LLM keyword-expansion hook is enabled.
    """

    context: str
    citations: list[str] = field(default_factory=list)
    usd: float = 0.0
    tokens: int = 0
    nodes_seen: int = 0


# --------------------------------------------------------------------------- #
# Search service                                                              #
# --------------------------------------------------------------------------- #
class SearchService:
    """Zero-LLM fulltext-seed + bounded-traversal search over the corpus graph.

    Pipeline:
      (a) rule-based keyword extraction -> Lucene OR query,
      (b) fulltext seed over ``corpus_ft`` -> seed Chunk ids,
      (c) seed chunks -> owner nodes (incoming ``HAS_CHUNK``) -> BOUNDED
          variable-length walk across the corpus edges -> reached owner nodes,
      (d) gather chunk text of reached owners into a bounded context string,
          each owner emitting one citation token.
    """

    def __init__(self, graph, cfg: dict | None = None) -> None:
        """Args:
        graph: injected ``Neo4jGraph``-like object exposing
            ``.query(cypher: str, params: dict) -> list[dict]``. We do NOT
            construct the driver — the adapter injects a graph scoped to the
            current candidate's ``graph_<hash>`` database.
        cfg: optional tunable overrides (see the ``DEFAULT_*`` constants).
        """
        cfg = cfg or {}
        self.graph = graph

        # --- headline tunables ------------------------------------------------
        self.seed_limit: int = int(cfg.get("seed_limit", DEFAULT_SEED_LIMIT))
        self.neighbor_limit: int = int(cfg.get("neighbor_limit", DEFAULT_NEIGHBOR_LIMIT))
        self.max_context_chars: int = int(cfg.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS))
        self.max_chunks_total: int = int(cfg.get("max_chunks_total", DEFAULT_MAX_CHUNKS_TOTAL))

        # max_depth is ALWAYS bounded: clamp into [1, depth_ceiling] so the walk
        # can never become an unbounded ``*`` traversal.
        depth_ceiling = int(cfg.get("depth_ceiling", DEFAULT_DEPTH_CEILING))
        requested_depth = int(cfg.get("max_depth", DEFAULT_MAX_DEPTH))
        self.max_depth: int = max(1, min(requested_depth, max(1, depth_ceiling)))

        # --- DISABLED LLM hook (off by default; keeps seed at usd=0/tokens=0) -
        # The optimizer may later flip this on and supply a chat model to add an
        # LLM keyword/expansion step. While False, NO LLM call is made.
        self.enable_llm_keyword_expansion: bool = bool(
            cfg.get("enable_llm_keyword_expansion", False)
        )
        self._chat_model = cfg.get("chat_model")  # only used if the hook is enabled

    # ------------------------------------------------------------------ #
    # Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def search(self, query: str) -> SearchResult:
        """Run the full seed -> bounded-walk -> context pipeline for *query*."""
        usd_total = 0.0
        tokens_total = 0

        keywords = self._extract_keywords(query)

        # DISABLED-by-default LLM expansion hook. Returns extra keywords AND the
        # usd/tokens it metered (both 0 while disabled). See _llm_expand_keywords.
        extra_keywords, hook_usd, hook_tokens = self._llm_expand_keywords(query, keywords)
        keywords = keywords + extra_keywords
        usd_total += hook_usd
        tokens_total += hook_tokens

        logger.info("[search] %d keyword(s) extracted: %s", len(keywords), keywords)
        if not keywords:
            return SearchResult(context="", citations=[], usd=usd_total, tokens=tokens_total)

        lucene_q = self._build_lucene_query(keywords)

        seed_ids = self._fulltext_seed(lucene_q)
        logger.info("[search] fulltext seed returned %d chunk(s)", len(seed_ids))
        if not seed_ids:
            return SearchResult(context="", citations=[], usd=usd_total, tokens=tokens_total)

        owners = self._bounded_walk(seed_ids)
        logger.info("[search] bounded walk (depth<=%d) reached %d owner node(s)",
                    self.max_depth, len(owners))
        if not owners:
            return SearchResult(
                context="", citations=[], usd=usd_total, tokens=tokens_total, nodes_seen=0,
            )

        context, citations = self._build_context(owners)
        logger.info("[search] built %d-char context with %d citation(s)",
                    len(context), len(citations))

        return SearchResult(
            context=context,
            citations=citations,
            usd=usd_total,
            tokens=tokens_total,
            nodes_seen=len(owners),
        )

    # ------------------------------------------------------------------ #
    # (a) Rule-based keyword extraction (ZERO LLM)                         #
    # ------------------------------------------------------------------ #
    def _extract_keywords(self, query: str) -> list[str]:
        """Tokenize *query* into search keywords WITHOUT any LLM call.

        Steps: preserve typed tokens (ticket keys / @handles) verbatim, then
        lowercase + split on non-alphanumerics, drop stopwords and tokens
        shorter than ``MIN_TOKEN_LEN``. De-duplicated, order preserved
        (typed tokens first).
        """
        if not query:
            return []

        # 1) Typed tokens preserved verbatim (case-sensitive), e.g. PROJ-12, @alice.
        typed: list[str] = []
        for pattern in _TYPED_TOKEN_PATTERNS:
            typed.extend(pattern.findall(query))

        # 2) General tokenization: lowercase, split on non-alphanumerics.
        raw_tokens = re.split(r"[^a-z0-9]+", query.lower())
        general = [
            tok for tok in raw_tokens
            if len(tok) >= MIN_TOKEN_LEN and tok not in _STOPWORDS
        ]

        # 3) De-dup (case-insensitive), typed tokens first, order preserved.
        ordered: list[str] = []
        seen: set[str] = set()
        for tok in typed + general:
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(tok)
        return ordered

    def _build_lucene_query(self, keywords: list[str]) -> str:
        """Build a Lucene ``OR`` query string with every term escaped.

        Escaping matters for typed tokens: ``PROJ-12`` must not be parsed as
        ``PROJ`` AND NOT ``12`` (a bare ``-`` is the Lucene NOT operator).
        """
        escaped = [_LUCENE_SPECIAL.sub(r"\\\1", kw) for kw in keywords if kw]
        return " OR ".join(escaped)

    # ------------------------------------------------------------------ #
    # DISABLED LLM hook (the ONE place an LLM step may later be added)     #
    # ------------------------------------------------------------------ #
    def _llm_expand_keywords(
        self, query: str, keywords: list[str]
    ) -> tuple[list[str], float, int]:
        """OPTIONAL LLM keyword/expansion step — DISABLED by default.

        While ``enable_llm_keyword_expansion`` is False (the default) this is a
        no-op returning ``([], 0.0, 0)`` so the seed stays at ``usd=0`` /
        ``tokens=0``. If the optimizer enables it, this is the single seam where
        an LLM call would live AND where it MUST meter and return ``(extra_keywords,
        usd, tokens)`` so the cost flows into ``SearchResult``.
        """
        if not self.enable_llm_keyword_expansion or self._chat_model is None:
            return [], 0.0, 0

        # --- intentionally not implemented in the zero-LLM seed --------------
        # An enabled implementation would, e.g.:
        #   resp = self._chat_model.invoke(<expansion prompt over `query`>)
        #   extra = _parse_keywords(resp.content)
        #   usd, tokens = _meter(resp)          # read usage off the response
        #   return extra, usd, tokens
        # Until that is wired in we refuse to silently pretend it ran.
        raise NotImplementedError(
            "LLM keyword expansion hook is enabled but not implemented; "
            "wire up self._chat_model and metering before enabling."
        )

    # ------------------------------------------------------------------ #
    # (b) Fulltext seed over `corpus_ft`                                   #
    # ------------------------------------------------------------------ #
    def _fulltext_seed(self, lucene_q: str) -> list[str]:
        """Return seed Chunk ids via the ``corpus_ft`` fulltext index.

        Parameterized; the index name is the contract constant ``corpus_ft``.
        Forced ``LIMIT $seed_limit`` keeps the seed bounded.
        """
        cypher = (
            "CALL db.index.fulltext.queryNodes('corpus_ft', $q) YIELD node, score "
            "RETURN node.id AS id, score "
            "ORDER BY score DESC "
            "LIMIT $seed_limit"
        )
        rows = self.graph.query(cypher, {"q": lucene_q, "seed_limit": self.seed_limit})
        return [r["id"] for r in rows if r.get("id")]

    # ------------------------------------------------------------------ #
    # (c) Bounded variable-length walk to reachable owner nodes           #
    # ------------------------------------------------------------------ #
    def _bounded_walk(self, seed_chunk_ids: list[str]) -> list[dict]:
        """From seed chunks, hop to owner nodes (incoming ``HAS_CHUNK``) then do a
        BOUNDED variable-length walk across the corpus edges to gather connected
        owners (Document/Message/Ticket).

        Returns owner dicts ``{"id": str, "labels": list[str]}`` in a stable,
        de-duplicated order (seed owners first thanks to the ``*0..N`` lower bound).

        NOTE on the hop bound: Cypher does not allow the variable-length range to
        be a query parameter, so ``self.max_depth`` (an int already clamped into
        ``[1, depth_ceiling]``) is formatted into the pattern as ``*0..<depth>``.
        It is ALWAYS bounded — never an open-ended ``*``. Everything else stays
        parameterized and every query carries a forced LIMIT.
        """
        edge_alt = "|".join(_CORPUS_EDGE_TYPES)  # AUTHORED_BY|ABOUT|REFERENCES|...
        depth = int(self.max_depth)               # already clamped; int-coerced for safety
        owner_label_pred = " OR ".join(f"reached:{lbl}" for lbl in _OWNER_LABELS)

        cypher = (
            "MATCH (chunk:Chunk) WHERE chunk.id IN $seed_ids "
            # seed chunks -> their owner node via the incoming HAS_CHUNK edge
            "MATCH (chunk)<-[:HAS_CHUNK]-(owner) "
            f"WHERE {' OR '.join(f'owner:{lbl}' for lbl in _OWNER_LABELS)} "
            "WITH collect(DISTINCT owner) AS seeds "
            "UNWIND seeds AS seed "
            # BOUNDED walk: *0..<depth> includes the seed owner itself (0 hops)
            f"MATCH (seed)-[:{edge_alt}*0..{depth}]-(reached) "
            f"WHERE {owner_label_pred} "
            "RETURN DISTINCT reached.id AS id, labels(reached) AS labels "
            "LIMIT $neighbor_limit"
        )
        rows = self.graph.query(
            cypher,
            {"seed_ids": seed_chunk_ids, "neighbor_limit": self.neighbor_limit},
        )
        return [
            {"id": r["id"], "labels": r.get("labels") or []}
            for r in rows
            if r.get("id")
        ]

    # ------------------------------------------------------------------ #
    # (d) Gather content + citations into a bounded context string        #
    # ------------------------------------------------------------------ #
    def _build_context(self, owners: list[dict]) -> tuple[str, list[str]]:
        """Gather chunk text of reached owners into a bounded context string.

        Each owner emits exactly one citation token (``[doc:<id>]`` /
        ``[msg:<id>]`` / ``[ticket:<id>]``). Bounded by both
        ``max_context_chars`` and ``max_chunks_total``.
        """
        owner_ids = [o["id"] for o in owners]
        labels_by_id: dict[str, list[str]] = {o["id"]: o["labels"] for o in owners}

        # Fetch all chunks for the reached owners in one bounded query.
        cypher = (
            "MATCH (owner)-[:HAS_CHUNK]->(c:Chunk) "
            "WHERE owner.id IN $owner_ids "
            "RETURN owner.id AS owner_id, c.id AS chunk_id, c.content AS content "
            "ORDER BY owner.id, c.id "
            "LIMIT $max_chunks_total"
        )
        rows = self.graph.query(
            cypher,
            {"owner_ids": owner_ids, "max_chunks_total": self.max_chunks_total},
        )

        # Group chunk contents by owner id (preserve owner order from the walk).
        content_by_owner: dict[str, list[str]] = {}
        for r in rows:
            oid = r.get("owner_id")
            content = (r.get("content") or "").strip()
            if not oid or not content:
                continue
            content_by_owner.setdefault(oid, []).append(content)

        parts: list[str] = []
        citations: list[str] = []
        used_chars = 0

        for oid in owner_ids:  # stable order: seed owners first
            chunks = content_by_owner.get(oid)
            if not chunks:
                continue  # owner reached but carries no chunk text -> no citation

            citation = self._citation_for(oid, labels_by_id.get(oid, []))
            block = citation + "\n" + "\n".join(chunks)

            # Bound total context length; stop cleanly once we would overflow.
            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= len(citation) + 1:
                    break  # no room for any meaningful content from this owner
                block = block[:remaining]
                parts.append(block)
                citations.append(citation)
                used_chars += len(block)
                break

            parts.append(block)
            citations.append(citation)
            used_chars += len(block) + 1  # +1 for the joining newline

        context = "\n".join(parts)
        return context, citations

    @staticmethod
    def _citation_for(owner_id: str, labels: list[str]) -> str:
        """Build the citation token for an owner node from its TYPE.

        The canonical ids already carry their prefix (``doc:`` / ``msg:`` /
        ``ticket:``); we key off the label so the prefix is authoritative and
        normalize the id so the token is exactly ``[doc:<local>]`` etc. — the
        ``retrieval_hit`` probe substring-matches the gold node id against this.
        """
        prefix = next(
            (_CITATION_PREFIX_BY_LABEL[lbl] for lbl in labels if lbl in _CITATION_PREFIX_BY_LABEL),
            None,
        )
        if prefix is None:
            # Unknown/owner-less label: emit the raw id verbatim (still bracketed).
            return f"[{owner_id}]"
        local = owner_id[len(prefix) + 1:] if owner_id.startswith(f"{prefix}:") else owner_id
        return f"[{prefix}:{local}]"


# --------------------------------------------------------------------------- #
# Module-level convenience entrypoint                                          #
# --------------------------------------------------------------------------- #
def run_search(query: str, graph, cfg: dict | None = None) -> SearchResult:
    """Construct a :class:`SearchService` and run a single *query*.

    Convenience wrapper the reward adapter can import and call directly.
    """
    return SearchService(graph, cfg).search(query)
