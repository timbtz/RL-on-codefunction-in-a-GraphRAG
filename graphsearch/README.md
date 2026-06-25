# graphsearch

Self-contained, in-repo copy of the **agentic graph-traversal search service** — the retrieval code the graphretr optimizer evolves. Copied out of `knowledge-management-platform` so this repo has no cross-repo dependency.

## Provenance

Copied (verbatim) from `/Users/t.betz/knowledge-management-platform/src` on 2026-06-23. Dependency closure was traced to be minimal and contamination-free: **no** dependency-injector container, MCP server, AWS, pollers, or ingestion pipeline. The package layout (`common/...`, `llm_graph_hybrid_vector_rag/...`) is preserved so the original absolute imports resolve unchanged.

## What's here

```
src/common/service/search/
  base/search_service.py                                  # SearchService ABC — the port shape
  agentic_graph_traversal_search_service.py               # PRIMARY evolve target (fulltext + recursive traversal, no embeddings)
  agentic_with_attribute_graph_traversal_search_service.py# variant (Document→Chunk→Entity schema); a second target
src/common/factory/progress_reporter/                     # NoopProgressReporter + base (only progress dep)
src/llm_graph_hybrid_vector_rag/service/rag/rag_system.py # CompanyRAGChain — mirror for answer synthesis
data/gold_qa.json                                         # 11 German Q&A pairs {question, answer, source}
```

## Construction (no DI container)

The services take fully-constructed objects — build them directly:

```python
import sys; sys.path.insert(0, "graphsearch/src")
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from common.service.search.agentic_graph_traversal_search_service import AgenticGraphTraversalSearchService

graph = Neo4jGraph(url=..., username=..., password=..., database=...)  # creds from env
chat  = ChatOpenAI(model="gpt-4o", temperature=0)                       # temperature=0 to cut reward noise
svc   = AgenticGraphTraversalSearchService(graph=graph, chat_model=chat)
context = svc.search("...")   # -> retrieved context with [doc:ID] citations
```

Requires a running Neo4j holding the ingested KG with a `ft_Entities` fulltext index. The `graph` and `chat_model` are constructor-injected, so the optimizer's eval harness wraps them with cost-metering proxies.

See the implementation plan: `../.claude/plan/agentic-search-optimizer-integration.md`.
