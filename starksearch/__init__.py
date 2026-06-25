"""starksearch -- the STaRK-prime search service the optimizer evolves.

Sibling of `graphsearch/` (the company-KG service): both are mutated as
`FileSet`s and scored through the engine's `SearchTarget` subprocess port. This
package holds the `G` retrieval service (`graph.py`), its closed primitive DSL
(`primitives.py`, `cache.py`, `embedder.py`), the FalkorDB backend
(`backends/`), the candidate `search(q, G)` seeds (`seeds/`), the STaRK
question-set loader (`qa/`), and the recall@20/hit@1/mrr scorer (`reward/`).

Carved out of `graphretr_opt.env` / `.data` / `.reward` (2026-06-25) so the
optimizer engine stays target-agnostic. The reward framework (MetricVector,
pareto dominance) and the eval seam (`search_target`, subprocess targets) stay
in the engine; only the STaRK-specific service code lives here.
"""
