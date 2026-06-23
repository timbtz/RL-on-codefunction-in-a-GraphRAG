import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from common.factory.progress_reporter.noop_progress_reporter import NoopProgressReporter
from common.factory.progress_reporter.progress_reporter_base import ProgressReporterBase
from langchain_neo4j import Neo4jGraph
from langchain.chat_models import BaseChatModel
from common.service.search.base.search_service import SearchService


class AgenticGraphTraversalSearchService(SearchService):
    def __init__(
        self,
        graph: Neo4jGraph,
        chat_model: BaseChatModel,
        max_depth: int = 3,
        neighbor_limit: int = 25,
        docs_per_entity: int = 25,
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
        self.docs_per_entity = docs_per_entity
        self.max_context_chars = max_context_chars
        self.max_frontier = max_frontier
        self.max_workers = max_workers
        self.fulltext_index_name = fulltext_index_name
        self.reporter: ProgressReporterBase = NoopProgressReporter()

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> str:
        start = time.perf_counter()
        logging.info("[SEARCH] Starting search for query: %s", query[:120])
        keywords = self._extract_keywords(query)
        logging.info("[SEARCH] Extracted keywords: %s", keywords)
        seed_nodes = self._fulltext_search(keywords)
        logging.info("[SEARCH] Fulltext search returned %d seed nodes", len(seed_nodes))
        result = self._agent_search(question=query, nodes=seed_nodes)
        elapsed = time.perf_counter() - start
        logging.info("[SEARCH] Search complete — %d chars in %.2fs", len(result), elapsed)
        return result

    async def asearch(self, query: str, reporter: ProgressReporterBase = NoopProgressReporter()) -> str:
        self.reporter = reporter
        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        await self.reporter.on_info("⏳ Extracting keywords...")
        keywords = await loop.run_in_executor(None, self._extract_keywords, query)
        seed_nodes = await loop.run_in_executor(None, self._fulltext_search, keywords)
        await self.reporter.on_info(f"🤖 Running search agent with {len(seed_nodes)} seed nodes...")
        result = await self._async_agent_search(question=query, nodes=seed_nodes)
        elapsed = time.perf_counter() - start
        await self.reporter.on_info(f"🚀 Finished search in {elapsed:.1f}s")
        return result

    # ------------------------------------------------------------------ #
    # Graph queries                                                        #
    # ------------------------------------------------------------------ #

    def _fulltext_search(self, keywords: str) -> list[str]:
        """Return entity IDs matching *keywords* via the fulltext index."""
        # Use a parameter to avoid Cypher injection.
        result = self.graph.query(
            f"""
            CALL db.index.fulltext.queryNodes("{self.fulltext_index_name}", $keywords)
            YIELD node, score
            RETURN node.id AS id
            ORDER BY score DESC
            """,
            {"keywords": keywords},
        )
        return [r["id"] for r in result if r.get("id")]

    def _get_neighbors(self, node_id: str) -> list[str]:
        """Return IDs of non-Document neighbours of *node_id*."""
        rows = self.graph.query(
            """
            MATCH (e {id: $id})-[r]-(n)
            WHERE n.id IS NOT NULL AND NOT n:Document
            RETURN n.id AS id
            LIMIT $limit
            """,
            {"id": node_id, "limit": self.neighbor_limit},
        )
        return [r["id"] for r in rows if r.get("id")]

    def _get_documents_for_entity(self, node_id: str) -> list[dict]:
        """Return Document nodes connected to *node_id*."""
        return self.graph.query(
            """
            MATCH (e {id: $id})-[r]-(d:Document)
            RETURN d.id AS id, d.text AS text, type(r) AS rel_type
            LIMIT $limit
            """,
            {"id": node_id, "limit": self.docs_per_entity},
        )

    # ------------------------------------------------------------------ #
    # LLM helpers                                                          #
    # ------------------------------------------------------------------ #

    def _invoke(self, system: str, user: str) -> str:
        """Single-turn LLM call; returns the text content."""
        res = self.chat_model.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        # LangChain chat models return an AIMessage; .content is the str.
        return res.content.strip()

    def _invoke_json(self, system: str, user: str, fallback: dict) -> dict:
        """LLM call that must return JSON; returns *fallback* on parse error."""
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

    def _is_document_useful(self, question: str, document_text: str) -> bool:
        answer = self._invoke(
            system=(
                "Evaluate whether the document content is useful for answering "
                "the question. Reply with exactly 'useful' or 'not useful'."
            ),
            user=json.dumps(
                {"question": question, "document_text": document_text},
                ensure_ascii=False,
            ),
        )
        return answer.lower().startswith("useful")

    def _judge_sufficient(self, question: str, context: str) -> dict:
        """Returns ``{"sufficient": bool, "missing": [str]}``."""
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

    def _pick_next_entities(self, question: str, context: str, candidates: list[str]) -> list[str]:
        """Ask the LLM which candidate entity IDs to explore next."""
        data = self._invoke_json(
            system=(
                "Select which entity node IDs to explore next to gather more relevant documents. "
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
                "Keep citations like [doc:ID] if present. Be concise."
            ),
            user=f"Question:\n{question}\n\nContext:\n{context}",
        )
        return compressed[-self.max_context_chars:]

    def _filter_useful_docs(
        self,
        question: str,
        docs: list[dict],
        max_chars_per_doc: int = 800,
    ) -> list[dict]:
        """
        Given a list of document dicts (with 'id' and 'text'), ask the LLM
        in ONE call which ones are relevant.  Returns only the useful docs.
        """
        if not docs:
            return []

        listing_parts: list[str] = []
        for idx, doc in enumerate(docs):
            text = (doc.get("text") or "").strip()
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc] + "…"
            listing_parts.append(f"[{idx}] (id={doc['id']})\n{text}")

        listing = "\n---\n".join(listing_parts)

        result = self._invoke_json(
            system=(
                "You are a relevance judge. Given a question and a numbered list "
                "of documents, return the **indices** (0-based) of every document "
                "that contains information useful for answering the question.\n\n"
                "Return ONLY valid JSON: {\"useful\": [0, 3, 7]}\n"
                "If none are useful return {\"useful\": []}."
            ),
            user=f"Question:\n{question}\n\nDocuments:\n{listing}",
            fallback={"useful": list(range(len(docs)))},
        )

        useful_indices = set(result.get("useful", []))
        return [docs[i] for i in sorted(useful_indices) if i < len(docs)]

    # ------------------------------------------------------------------ #
    # Per-node work (parallelised)                                         #
    # ------------------------------------------------------------------ #

    def _process_node(
        self,
        node_id: str,
        question: str,
        visited_docs: frozenset,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """
        Fetch documents + neighbours for a single node.
        Returns ([(doc_id, formatted_text), ...], [neighbour_id, ...]).
        """
        raw_docs: list[dict] = []
        for doc in self._get_documents_for_entity(node_id):
            doc_id = doc.get("id")
            if not doc_id or doc_id in visited_docs:
                continue
            text = (doc.get("text") or "").strip()
            if text:
                raw_docs.append({"id": doc_id, "text": text})

        useful_docs = self._filter_useful_docs(question, raw_docs)

        useful_texts = [
            (d["id"], f"[doc:{d['id']}]\n{d['text']}\n")
            for d in useful_docs
        ]

        neighbour_ids = self._get_neighbors(node_id)

        return useful_texts, neighbour_ids

    # ------------------------------------------------------------------ #
    # Core agentic search (recursive)                                      #
    # ------------------------------------------------------------------ #

    def _agent_search(
        self,
        question: str,
        nodes: list[str],
        text_context: str = "",
        visited_nodes: set | None = None,
        visited_docs: set | None = None,
        depth: int = 0,
    ) -> str:
        if visited_nodes is None:
            visited_nodes = set()
        if visited_docs is None:
            visited_docs = set()

        frontier = list(dict.fromkeys(
            n for n in nodes if n and n not in visited_nodes
        ))[:self.max_frontier]

        if not frontier:
            logging.info("[AGENT depth=%d] No frontier nodes — stopping.", depth)
            return text_context
        if depth > self.max_depth:
            logging.info("[AGENT depth=%d] Max depth reached — stopping.", depth)
            return text_context

        logging.info("[AGENT depth=%d] Processing %d frontier nodes", depth, len(frontier))

        new_text_parts: list[str] = []
        next_candidates: list[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_node, node_id, question, frozenset(visited_docs)): node_id
                for node_id in frontier
            }
            for future in as_completed(futures):
                node_id = futures[future]
                visited_nodes.add(node_id)
                try:
                    useful_texts, neighbour_ids = future.result()
                except Exception:
                    logging.exception("Error processing node %s", node_id)
                    continue

                for doc_id, text_part in useful_texts:
                    if doc_id not in visited_docs:
                        visited_docs.add(doc_id)
                        new_text_parts.append(text_part)

                for nid in neighbour_ids:
                    if nid not in visited_nodes:
                        next_candidates.append(nid)

        logging.info(
            "[AGENT depth=%d] Collected %d new docs, %d neighbour candidates",
            depth, len(new_text_parts), len(next_candidates),
        )
        if new_text_parts:
            text_context = (text_context + "\n" + "\n".join(new_text_parts)).strip()

        if len(text_context) > self.max_context_chars:
            logging.info("[AGENT depth=%d] Context too large (%d chars) — compressing", depth, len(text_context))
            text_context = self._compress_context(question, text_context)

        judgement = self._judge_sufficient(question, text_context)
        if judgement.get("sufficient"):
            logging.info("[AGENT depth=%d] Context judged sufficient — done.", depth)
            return text_context

        logging.info("[AGENT depth=%d] Context not yet sufficient, missing: %s", depth, judgement.get("missing"))

        unique_candidates = list(dict.fromkeys(
            x for x in next_candidates if x not in visited_nodes
        ))
        if not unique_candidates:
            logging.info("[AGENT depth=%d] No more candidates to explore — stopping.", depth)
            return text_context

        chosen = self._pick_next_entities(question, text_context, unique_candidates)
        chosen = [c for c in chosen if c in set(unique_candidates)] or unique_candidates[:5]

        logging.info("[AGENT depth=%d] Exploring %d next entities", depth, len(chosen))

        return self._agent_search(
            question=question,
            nodes=chosen,
            text_context=text_context,
            visited_nodes=visited_nodes,
            visited_docs=visited_docs,
            depth=depth + 1,
        )

    async def _async_agent_search(
        self,
        question: str,
        nodes: list[str],
        text_context: str = "",
        visited_nodes: set | None = None,
        visited_docs: set | None = None,
        depth: int = 0,
    ) -> str:
        loop = asyncio.get_event_loop()

        if visited_nodes is None:
            visited_nodes = set()
        if visited_docs is None:
            visited_docs = set()

        frontier = list(dict.fromkeys(
            n for n in nodes if n and n not in visited_nodes
        ))[:self.max_frontier]

        if not frontier:
            logging.info("[AGENT depth=%d] No frontier nodes — stopping.", depth)
            return text_context
        if depth > self.max_depth:
            logging.info("[AGENT depth=%d] Max depth reached — stopping.", depth)
            return text_context

        logging.info("[AGENT depth=%d] Processing %d frontier nodes", depth, len(frontier))
        await self.reporter.on_info(f"🔍 Searching the knowledge graph... (pass {depth + 1})")

        def _process_all() -> tuple[list[str], list[str]]:
            new_text_parts: list[str] = []
            next_candidates: list[str] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(self._process_node, node_id, question, frozenset(visited_docs)): node_id
                    for node_id in frontier
                }
                for future in as_completed(futures):
                    node_id = futures[future]
                    visited_nodes.add(node_id)
                    try:
                        useful_texts, neighbour_ids = future.result()
                    except Exception:
                        logging.exception("Error processing node %s", node_id)
                        continue
                    for doc_id, text_part in useful_texts:
                        if doc_id not in visited_docs:
                            visited_docs.add(doc_id)
                            new_text_parts.append(text_part)
                    for nid in neighbour_ids:
                        if nid not in visited_nodes:
                            next_candidates.append(nid)
            return new_text_parts, next_candidates

        new_text_parts, next_candidates = await loop.run_in_executor(None, _process_all)

        logging.info(
            "[AGENT depth=%d] Collected %d new docs, %d neighbour candidates",
            depth, len(new_text_parts), len(next_candidates),
        )
        await self.reporter.on_info(f"📄 Retrieved {len(new_text_parts)} relevant documents — checking if that's enough...")

        if new_text_parts:
            text_context = (text_context + "\n" + "\n".join(new_text_parts)).strip()

        if len(text_context) > self.max_context_chars:
            logging.info("[AGENT depth=%d] Context too large (%d chars) — compressing", depth, len(text_context))
            await self.reporter.on_info("🗜️ Compressing context...")
            text_context = await loop.run_in_executor(None, self._compress_context, question, text_context)

        judgement = await loop.run_in_executor(None, self._judge_sufficient, question, text_context)
        if judgement.get("sufficient"):
            logging.info("[AGENT depth=%d] Context judged sufficient — done.", depth)
            return text_context

        logging.info("[AGENT depth=%d] Context not yet sufficient, missing: %s", depth, judgement.get("missing"))
        await self.reporter.on_info("🔎 Need more context — going deeper...")

        unique_candidates = list(dict.fromkeys(
            x for x in next_candidates if x not in visited_nodes
        ))
        if not unique_candidates:
            logging.info("[AGENT depth=%d] No more candidates to explore — stopping.", depth)
            return text_context

        chosen = await loop.run_in_executor(None, self._pick_next_entities, question, text_context, unique_candidates)
        chosen = [c for c in chosen if c in set(unique_candidates)] or unique_candidates[:5]

        logging.info("[AGENT depth=%d] Exploring %d next entities", depth, len(chosen))

        return await self._async_agent_search(
            question=question,
            nodes=chosen,
            text_context=text_context,
            visited_nodes=visited_nodes,
            visited_docs=visited_docs,
            depth=depth + 1,
        )
