"""qa_eval -- the thin, importable surface the optimizer's subprocess worker
calls to build and run the real agentic search service over a single query.

This is the ONLY graphsearch module the optimizer depends on. It constructs the
service directly (no DI container; mirrors graphsearch/README.md) and optionally
wraps the Neo4j graph + chat model with cost-metering proxies so the eval seam
can report llm_calls / db_queries / tokens without touching the service code.
"""
