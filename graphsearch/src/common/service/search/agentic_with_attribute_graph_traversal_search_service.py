import asyncio
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.chat_models import BaseChatModel
from langchain_neo4j import Neo4jGraph

from common.factory.progress_reporter.noop_progress_reporter import NoopProgressReporter
from common.factory.progress_reporter.progress_reporter_base import ProgressReporterBase
from common.service.search.base.search_service import SearchService


class AgenticWithAttributeGraphTraversalSearchService(SearchService):
    """Agentic search for the Document -> Chunk -> Entity graph schema.

    NOTE: This search service must only be used together with
    LLMGraphIngestionWithSourceAttributeService. The graph traversal queries
    are tightly coupled to the schema that ingestion service produces
    (video-only at the moment):
        (:Document:Video)-[:HAS_CHUNK]->(:Chunk:VideoChunk)-[:MENTIONS]->(entity)

    Using this search service against a graph built by a different ingestion
    pipeline will likely return empty or incorrect results.

    The agent walks the entity graph, collects chunk texts that are
    reachable via MENTIONS, and uses an LLM to decide when it has
    gathered enough context to answer the question.
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        chat_model: BaseChatModel,
        max_depth: int = 3,
        neighbor_limit: int = 25,
        chunks_per_entity: int = 25,
        max_context_chars: int = 200_000,
        max_frontier: int = 40,
        max_workers: int = 20,
        fulltext_index_name: str = "ft_Entities",
    ):
        super().__init__()
        self.graph = graph
        self.chat_model = chat_model
        self.max_depth = max_depth
        self.neighbor_limit = neighbor_limit
        self.chunks_per_entity = chunks_per_entity
        self.max_context_chars = max_context_chars
        self.max_frontier = max_frontier
        self.max_workers = max_workers
        self.fulltext_index_name = fulltext_index_name
        self.reporter: ProgressReporterBase = NoopProgressReporter()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> str:
        start = time.perf_counter()
        logging.info("[SEARCH] query: %s", query[:120])

        keywords = self._extract_keywords(query)
        logging.info("[SEARCH] keywords: %s", keywords)

        seed_nodes = self._fulltext_search(keywords)
        logging.info("[SEARCH] %d seed nodes", len(seed_nodes))

        result = self._agent_search(question=query, nodes=seed_nodes)

        elapsed = time.perf_counter() - start
        logging.info("[SEARCH] done — %d chars in %.2fs", len(result), elapsed)
        return result

    async def asearch(
        self,
        query: str,
        reporter: ProgressReporterBase = NoopProgressReporter(),
    ) -> str:
        self.reporter = reporter
        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        await self.reporter.on_info("Extracting keywords…")
        keywords = await loop.run_in_executor(None, self._extract_keywords, query)

        seed_nodes = await loop.run_in_executor(None, self._fulltext_search, keywords)
        await self.reporter.on_info(f"Running search agent with {len(seed_nodes)} seed nodes…")

        result = await self._async_agent_search(question=query, nodes=seed_nodes)

        elapsed = time.perf_counter() - start
        await self.reporter.on_info(f"Finished search in {elapsed:.1f}s")
        return result

    # ------------------------------------------------------------------ #
    # Graph queries                                                        #
    # ------------------------------------------------------------------ #

    def _fulltext_search(self, keywords: str) -> list[str]:
        rows = self.graph.query(
            f"""
            CALL db.index.fulltext.queryNodes("{self.fulltext_index_name}", $keywords)
            YIELD node, score
            RETURN node.id AS id
            ORDER BY score DESC
            """,
            {"keywords": keywords},
        )
        return [r["id"] for r in rows if r.get("id")]

    def _get_neighbors(self, node_id: str) -> list[str]:
        """Return IDs of neighbouring entity nodes (skip Document and Chunk nodes)."""
        rows = self.graph.query(
            """
            MATCH (e {id: $id})-[r]-(n)
            WHERE n.id IS NOT NULL
              AND NOT n:Document
              AND NOT n:Chunk
            RETURN n.id AS id
            LIMIT $limit
            """,
            {"id": node_id, "limit": self.neighbor_limit},
        )
        return [r["id"] for r in rows if r.get("id")]

    def _get_chunks_for_entity(self, node_id: str) -> list[dict]:
        """Return Chunk nodes that MENTION this entity, plus source info."""
        return self.graph.query(
            """
            MATCH (e {id: $id})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
            RETURN c.id   AS chunk_id,
                   c.text  AS text,
                   d.id    AS source
            LIMIT $limit
            """,
            {"id": node_id, "limit": self.chunks_per_entity},
        )

    # ------------------------------------------------------------------ #
    # LLM helpers                                                          #
    # ------------------------------------------------------------------ #

    def _invoke(self, system: str, user: str) -> str:
        res = self.chat_model.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return res.content.strip()

    def _invoke_json(self, system: str, user: str, fallback: dict) -> dict:
        raw = self._invoke(system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("LLM returned non-JSON: %s", raw[:200])
            return fallback

    def _extract_keywords(self, query: str) -> str:
        return self._invoke(
            system=(
                "Extract the main keywords from the following user query to perform "
                "a graph search. Return only a list of keywords separated by ' OR '."
            ),
            user=query,
        )

    def _judge_sufficient(self, question: str, context: str) -> dict:
        return self._invoke_json(
            system=(
                "Decide whether the provided context is sufficient to answer the question. "
                "Return ONLY valid JSON with keys: sufficient (bool), missing (list of str)."
            ),
            user=json.dumps(
                {"question": question, "context": context},
                ensure_ascii=False,
            ),
            fallback={"sufficient": False, "missing": ["JSON_PARSE_ERROR"]},
        )

    def _pick_next_entities(
        self, question: str, context: str, candidates: list[str],
    ) -> list[str]:
        data = self._invoke_json(
            system=(
                "Select which entity node IDs to explore next to gather more relevant context. "
                "Use the question and current context. "
                "Return ONLY valid JSON: {\"next_node_ids\": [\"id1\", ...]}."
            ),
            user=json.dumps(
                {"question": question, "context": context, "candidate_node_ids": candidates},
                ensure_ascii=False,
            ),
            fallback={"next_node_ids": candidates[:5]},
        )
        chosen = data.get("next_node_ids") or []
        return [x for x in chosen if isinstance(x, str)]

    def _compress_context(self, question: str, context: str) -> str:
        compressed = self._invoke(
            system=(
                "Compress the context while preserving all facts needed to answer the question. "
                "Keep citations like [source:ID] if present. Be concise."
            ),
            user=f"Question:\n{question}\n\nContext:\n{context}",
        )
        return compressed[-self.max_context_chars:]

    def _filter_useful_chunks(
        self,
        question: str,
        chunks: list[dict],
        max_chars_per_chunk: int = 800,
    ) -> list[dict]:
        if not chunks:
            return []

        listing_parts: list[str] = []
        for idx, chunk in enumerate(chunks):
            text = (chunk.get("text") or "").strip()
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "…"
            listing_parts.append(
                f"[{idx}] (chunk={chunk['chunk_id']}, source={chunk.get('source', '?')})\n{text}"
            )

        listing = "\n---\n".join(listing_parts)

        result = self._invoke_json(
            system=(
                "You are a relevance judge. Given a question and a numbered list "
                "of text chunks, return the **indices** (0-based) of every chunk "
                "that contains information useful for answering the question.\n\n"
                "Return ONLY valid JSON: {\"useful\": [0, 3, 7]}\n"
                "If none are useful return {\"useful\": []}."
            ),
            user=f"Question:\n{question}\n\nChunks:\n{listing}",
            fallback={"useful": list(range(len(chunks)))},
        )

        useful_indices = set(result.get("useful", []))
        return [chunks[i] for i in sorted(useful_indices) if i < len(chunks)]

    # ------------------------------------------------------------------ #
    # Per-node work (parallelised)                                         #
    # ------------------------------------------------------------------ #

    def _process_node(
        self,
        node_id: str,
        question: str,
        visited_chunks: frozenset,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Fetch chunks + neighbours for a single entity node."""
        raw_chunks: list[dict] = []
        for row in self._get_chunks_for_entity(node_id):
            chunk_id = row.get("chunk_id")
            if not chunk_id or chunk_id in visited_chunks:
                continue
            text = (row.get("text") or "").strip()
            if text:
                raw_chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": row.get("source", "unknown"),
                })

        useful = self._filter_useful_chunks(question, raw_chunks)

        formatted = [
            (
                c["chunk_id"],
                f"[source:{c['source']}] [chunk:{c['chunk_id']}]\n{c['text']}\n",
            )
            for c in useful
        ]

        neighbour_ids = self._get_neighbors(node_id)
        return formatted, neighbour_ids

    # ------------------------------------------------------------------ #
    # Core agentic loop                                                    #
    # ------------------------------------------------------------------ #

    def _agent_search(
        self,
        question: str,
        nodes: list[str],
        text_context: str = "",
        visited_nodes: set | None = None,
        visited_chunks: set | None = None,
        depth: int = 0,
    ) -> str:
        if visited_nodes is None:
            visited_nodes = set()
        if visited_chunks is None:
            visited_chunks = set()

        frontier = list(dict.fromkeys(
            n for n in nodes if n and n not in visited_nodes
        ))[:self.max_frontier]

        if not frontier or depth > self.max_depth:
            logging.info("[AGENT depth=%d] Stopping (frontier=%d).", depth, len(frontier))
            return text_context

        logging.info("[AGENT depth=%d] Processing %d frontier nodes", depth, len(frontier))

        new_text_parts: list[str] = []
        next_candidates: list[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._process_node, nid, question, frozenset(visited_chunks),
                ): nid
                for nid in frontier
            }
            for future in as_completed(futures):
                nid = futures[future]
                visited_nodes.add(nid)
                try:
                    texts, neighbours = future.result()
                except Exception:
                    logging.exception("Error processing node %s", nid)
                    continue
                for chunk_id, text_part in texts:
                    if chunk_id not in visited_chunks:
                        visited_chunks.add(chunk_id)
                        new_text_parts.append(text_part)
                for nb in neighbours:
                    if nb not in visited_nodes:
                        next_candidates.append(nb)

        logging.info(
            "[AGENT depth=%d] %d new chunks, %d neighbour candidates",
            depth, len(new_text_parts), len(next_candidates),
        )

        if new_text_parts:
            text_context = (text_context + "\n" + "\n".join(new_text_parts)).strip()

        if len(text_context) > self.max_context_chars:
            text_context = self._compress_context(question, text_context)

        judgement = self._judge_sufficient(question, text_context)
        if judgement.get("sufficient"):
            logging.info("[AGENT depth=%d] Sufficient — done.", depth)
            return text_context

        logging.info("[AGENT depth=%d] Not sufficient, missing: %s", depth, judgement.get("missing"))

        unique_candidates = list(dict.fromkeys(
            x for x in next_candidates if x not in visited_nodes
        ))
        if not unique_candidates:
            return text_context

        chosen = self._pick_next_entities(question, text_context, unique_candidates)
        chosen = [c for c in chosen if c in set(unique_candidates)] or unique_candidates[:5]

        return self._agent_search(
            question=question,
            nodes=chosen,
            text_context=text_context,
            visited_nodes=visited_nodes,
            visited_chunks=visited_chunks,
            depth=depth + 1,
        )

    async def _async_agent_search(
        self,
        question: str,
        nodes: list[str],
        text_context: str = "",
        visited_nodes: set | None = None,
        visited_chunks: set | None = None,
        depth: int = 0,
    ) -> str:
        loop = asyncio.get_event_loop()

        if visited_nodes is None:
            visited_nodes = set()
        if visited_chunks is None:
            visited_chunks = set()

        frontier = list(dict.fromkeys(
            n for n in nodes if n and n not in visited_nodes
        ))[:self.max_frontier]

        if not frontier or depth > self.max_depth:
            return text_context

        logging.info("[AGENT depth=%d] Processing %d frontier nodes", depth, len(frontier))
        await self.reporter.on_info(f"Searching the knowledge graph… (pass {depth + 1})")

        def _process_all():
            new_text_parts: list[str] = []
            next_candidates: list[str] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self._process_node, nid, question, frozenset(visited_chunks),
                    ): nid
                    for nid in frontier
                }
                for future in as_completed(futures):
                    nid = futures[future]
                    visited_nodes.add(nid)
                    try:
                        texts, neighbours = future.result()
                    except Exception:
                        logging.exception("Error processing node %s", nid)
                        continue
                    for chunk_id, text_part in texts:
                        if chunk_id not in visited_chunks:
                            visited_chunks.add(chunk_id)
                            new_text_parts.append(text_part)
                    for nb in neighbours:
                        if nb not in visited_nodes:
                            next_candidates.append(nb)
            return new_text_parts, next_candidates

        new_text_parts, next_candidates = await loop.run_in_executor(None, _process_all)

        await self.reporter.on_info(
            f"Retrieved {len(new_text_parts)} relevant chunks — checking sufficiency…"
        )

        if new_text_parts:
            text_context = (text_context + "\n" + "\n".join(new_text_parts)).strip()

        if len(text_context) > self.max_context_chars:
            await self.reporter.on_info("Compressing context…")
            text_context = await loop.run_in_executor(
                None, self._compress_context, question, text_context,
            )

        judgement = await loop.run_in_executor(
            None, self._judge_sufficient, question, text_context,
        )
        if judgement.get("sufficient"):
            return text_context

        await self.reporter.on_info("Need more context — going deeper…")

        unique_candidates = list(dict.fromkeys(
            x for x in next_candidates if x not in visited_nodes
        ))
        if not unique_candidates:
            return text_context

        chosen = await loop.run_in_executor(
            None, self._pick_next_entities, question, text_context, unique_candidates,
        )
        chosen = [c for c in chosen if c in set(unique_candidates)] or unique_candidates[:5]

        return await self._async_agent_search(
            question=question,
            nodes=chosen,
            text_context=text_context,
            visited_nodes=visited_nodes,
            visited_chunks=visited_chunks,
            depth=depth + 1,
        )
