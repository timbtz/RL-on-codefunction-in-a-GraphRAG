"""starksearch -- the STaRK-prime search service the optimizer evolves.

Sibling of `graphsearch/` (the company-KG service): both are mutated as
`FileSet`s and scored through the engine's `SearchTarget` subprocess port. The
editable service lives under `src/` -- `src/stark_search/` (the
`StarkGraphSearchService` the optimizer rewrites) and `src/stark_harness/` (the
non-editable harness that injects + meters + firewalls FalkorDB/embedder/LLM).
Around it: the FalkorDB backend (`backends/`), the query embedder
(`embedder.py`), the STaRK question-set loader (`qa/`), and the
recall@20/hit@1/mrr scorer (`reward/`).

Carved out of `graphretr_opt.env` / `.data` / `.reward` (2026-06-25) so the
optimizer engine stays target-agnostic. The reward framework (MetricVector,
pareto dominance) and the eval seam (`search_target`, subprocess targets) stay in
the engine; only the STaRK-specific service code lives here. The old in-process
`G` DSL (`graph.py`/`primitives.py`) and AST sandbox were deleted in that move --
the candidate is now a whole editable service, not a sandboxed `search(q, G)`.
"""
