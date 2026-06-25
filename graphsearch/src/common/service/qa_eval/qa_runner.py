"""qa_runner -- build + run the real agentic search service for one query.

One deep function pair:
  build_service(neo4j_cfg, llm_cfg, instrument) -> AgenticGraphTraversalSearchService
  run_query(service, query) -> str

The service takes fully-constructed objects (Neo4jGraph + a chat model), so we
build them directly here -- no dependency-injector container (mirrors
graphsearch/README.md). When an `instrument` (CostSink) is supplied, the graph
and chat model are wrapped in transparent metering proxies so the optimizer can
read llm_calls / db_queries / token counts WITHOUT editing the service -- the
metering lives in the harness, the artifact stays clean.

Hard rules (eval hygiene + reward stability):
  * temperature=0 everywhere (cut reward noise).
  * read-only: never create/mutate the DB.
  * creds come from the caller (env-sourced); never hardcoded here.
"""
import logging
import os
import time
from dataclasses import dataclass, field

from langchain_neo4j import Neo4jGraph

from common.service.search.agentic_graph_traversal_search_service import (
    AgenticGraphTraversalSearchService,
)

# Map a relpath -> the service class it defines, so a future editable_files set
# (e.g. the attribute variant) can be selected by config without touching the
# worker. The default target is the primary agentic service.
_SERVICE_BY_RELPATH = {
    "common/service/search/agentic_graph_traversal_search_service.py":
        AgenticGraphTraversalSearchService,
}
DEFAULT_SERVICE_CLASS = AgenticGraphTraversalSearchService


@dataclass
class CostSink:
    """Mutable cost accumulator the metering proxies write into. One per
    build_service() call; the worker reads it after each query and resets."""
    llm_calls: int = 0
    db_queries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd_cost: float = 0.0          # REAL OpenRouter USD cost (token_usage.cost)

    def reset(self):
        self.llm_calls = self.db_queries = self.tokens_in = self.tokens_out = 0
        self.usd_cost = 0.0

    def snapshot(self) -> dict:
        return {"llm_calls": self.llm_calls, "db_queries": self.db_queries,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "usd_cost": self.usd_cost}


def _usd_from_response(res) -> float:
    """OpenRouter returns the REAL per-call USD cost in
    response_metadata['token_usage']['cost'] when usage accounting is enabled
    (extra_body={'usage': {'include': True}}). Returns 0.0 if absent (e.g. stock
    OpenAI, or accounting disabled) -- then only token counts are available."""
    try:
        meta = getattr(res, "response_metadata", None) or {}
        return float((meta.get("token_usage") or {}).get("cost") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


class _MeteredGraph:
    """Transparent proxy over Neo4jGraph that counts .query() calls. Everything
    else delegates to the wrapped graph (the service only calls .query)."""

    def __init__(self, graph, sink: CostSink):
        self._graph = graph
        self._sink = sink

    def query(self, *args, **kwargs):
        self._sink.db_queries += 1
        return self._graph.query(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._graph, name)


class _MeteredChat:
    """Transparent proxy over a chat model that counts .invoke() calls and, when
    the response carries usage_metadata, the token totals."""

    def __init__(self, chat, sink: CostSink):
        self._chat = chat
        self._sink = sink

    def invoke(self, *args, **kwargs):
        self._sink.llm_calls += 1
        res = self._chat.invoke(*args, **kwargs)
        usage = getattr(res, "usage_metadata", None) or {}
        try:
            self._sink.tokens_in += int(usage.get("input_tokens", 0) or 0)
            self._sink.tokens_out += int(usage.get("output_tokens", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            pass
        self._sink.usd_cost += _usd_from_response(res)
        return res

    def __getattr__(self, name):
        return getattr(self._chat, name)


def _openai_compatible_kwargs():
    """Route OpenAI-family models through an OpenAI-compatible gateway (OpenRouter)
    when OPENROUTER_API_KEY is set; otherwise return {} so ChatOpenAI falls back to
    the stock OpenAI client (OPENAI_API_KEY). The subprocess worker inherits the
    parent env, so this reads the same OPENROUTER_* vars the optimizer was run with."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {}
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # usage.include makes OpenRouter return the REAL USD cost per call in
    # response_metadata.token_usage.cost (read by _usd_from_response).
    return {"base_url": base, "api_key": key,
            "extra_body": {"usage": {"include": True}}}


class _CliMsg:
    """Minimal AIMessage stand-in: the search service + McqReward only read
    `.content`; the metering proxy probes `.usage_metadata`/`.response_metadata`
    (absent for CLI -> 0 cost, which is correct: CLI bills the subscription)."""
    def __init__(self, content):
        self.content = content
        self.usage_metadata = None
        self.response_metadata = {}


class ClaudeCliChat:
    """Chat model that routes `.invoke()` through the authenticated Claude CLI
    (`claude -p --model`), so in-search + answerer LLM calls bill the Claude
    subscription instead of OpenRouter. Has a hard per-call subprocess timeout +
    retries (a stalled call can't wedge the run). Raises on repeated failure /
    rate-limit so the caller records it as a per-query miss."""
    def __init__(self, model="claude-haiku-4-5", timeout_s=120, max_retries=2):
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @staticmethod
    def _strip_fences(s):
        """Drop ```json ... ``` / ``` ... ``` markdown fences (Claude often wraps
        JSON/code even when told not to) so `json.loads` on the result works."""
        s = s.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    @staticmethod
    def _to_prompt(messages):
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content", "")
            else:
                role, content = getattr(m, "type", "user"), getattr(m, "content", "")
            parts.append(content if role == "user" else f"[{role}]\n{content}")
        return "\n\n".join(str(p) for p in parts)

    # Reframe the Claude Code CLI as a plain completion engine -- otherwise it
    # answers AS the coding agent ("this is Claude Code, not a GraphRAG
    # interface...") and breaks the search's JSON/keyword prompts.
    _SYS = ("You are a non-interactive text/JSON completion engine embedded in an "
            "automated data pipeline. The message is a task from a retrieval "
            "system. Follow it literally and output ONLY the exact requested "
            "content (e.g. the JSON object, or the keyword list) -- no preamble, "
            "no questions, no explanation, no markdown code fences. If asked for "
            "JSON, output only valid JSON.")

    def invoke(self, messages, *args, **kwargs):
        import subprocess
        prompt = self._to_prompt(messages)
        last = "no attempt"
        for _ in range(self.max_retries + 1):
            try:
                r = subprocess.run(
                    ["claude", "-p", "--model", self.model,
                     "--append-system-prompt", self._SYS],
                    input=prompt, capture_output=True, text=True,
                    timeout=self.timeout_s)
                if r.returncode == 0 and r.stdout.strip():
                    return _CliMsg(self._strip_fences(r.stdout))
                last = f"rc={r.returncode}: {(r.stderr or r.stdout)[:200]}"
            except subprocess.TimeoutExpired:
                last = f"cli timeout >{self.timeout_s}s"
        raise RuntimeError(f"claude CLI failed after {self.max_retries + 1} tries: {last}")


def _build_chat_model(llm_cfg):
    """Construct the chat model from llm_cfg = {provider, model, temperature}.
    temperature is forced to 0 regardless of the cfg (reward stability)."""
    provider = (llm_cfg.get("provider") or "openai").lower()
    model = llm_cfg["model"]
    if provider in ("claude_cli", "cli"):
        return ClaudeCliChat(model=model,
                             timeout_s=int(llm_cfg.get("timeout_s", 120) or 120))
    if provider in ("openai", "azure_openai"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, request_timeout=60,
                          max_retries=2, **_openai_compatible_kwargs())
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, timeout=60, max_retries=2)
    raise ValueError(f"unknown llm provider {provider!r}")


def build_service(neo4j_cfg, llm_cfg, instrument: CostSink | None = None,
                  service_relpath: str | None = None,
                  service_kwargs: dict | None = None):
    """Construct AgenticGraphTraversalSearchService (or another configured
    target) with a Neo4jGraph + chat model, optionally metered.

    neo4j_cfg: {url, username, password, database}
    llm_cfg:   {provider, model[, temperature]}  -- temperature is forced to 0
    instrument: a CostSink to meter into (None = no metering)
    service_relpath: which editable target class to build (defaults to the
                     primary agentic service)
    service_kwargs: extra ctor kwargs for the service (e.g. fulltext_index_name)
    """
    graph = Neo4jGraph(
        url=neo4j_cfg["url"],
        username=neo4j_cfg["username"],
        password=neo4j_cfg["password"],
        database=neo4j_cfg.get("database", "neo4j"),
    )
    chat = _build_chat_model(llm_cfg)
    if instrument is not None:
        graph = _MeteredGraph(graph, instrument)
        chat = _MeteredChat(chat, instrument)
    cls = _SERVICE_BY_RELPATH.get(service_relpath, DEFAULT_SERVICE_CLASS)
    return cls(graph=graph, chat_model=chat, **(service_kwargs or {}))


def run_query(service, query: str) -> str:
    """Run one query through the service synchronously, returning the retrieved
    context string. Errors propagate to the caller (the worker serializes them
    as a per-query crash)."""
    start = time.perf_counter()
    try:
        return service.search(query)
    finally:
        logging.info("[qa_runner] query done in %.2fs", time.perf_counter() - start)
