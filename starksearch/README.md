# starksearch

Self-contained STaRK-prime **graph search service** — the retrieval code the
`graphretr_opt` optimizer evolves for the STaRK biomedical benchmark. Sibling of
[`graphsearch/`](../graphsearch/README.md) (the company-KG service); both are
mutated as `FileSet`s and scored through the same engine `SearchTarget`
subprocess port.

## Provenance

Carved out of `graphretr-demo/src/graphretr_opt` (2026-06-25) so the optimizer
engine is target-agnostic. Moved verbatim (imports repointed):

```
graph.py            ← env/retrieval_graph.py   # the `G` service the candidate sees
primitives.py       ← env/primitives.py        # closed primitive DSL + allowlists + caps
cache.py            ← env/cache.py
embedder.py         ← env/embedder.py
backends/base.py    ← env/backends/base.py      # GraphBackend ABC
backends/falkordb.py← env/backends/falkordb.py  # read-only FalkorDB backend
seeds/*.py          ← artifact/seeds/*.py        # candidate search(q, G) programs
qa/substrate.py     ← data/substrate.py          # STaRK question-set loader (Substrate)
qa/loader_etl.py    ← data/loader_etl.py
reward/evaluator.py ← reward/evaluator.py        # recall@20/hit@1/mrr scoring (RewardModel)
```

## Contract

The candidate program is `def search(q, G) -> dict[int, float]` (node id →
score). `G` is a `starksearch.graph.RetrievalGraph` over a read-only FalkorDB
holding the STaRK-prime KG. The service enforces, in `primitives.py`, the caps
that protect the *shared* FalkorDB container — read-only backend, row caps, and
the **`algo.SPpaths maxLen ≤ 3` wall** (`maxLen ≥ 4` ignores the query timeout
and pegs the container). These guards survive the loosening of the old AST
sandbox: a subprocess wall-clock kill stops a runaway Python candidate but NOT a
runaway server-side query.

## Packaging

Lightweight import shim (like `graphsearch`): the package lives at the monorepo
root and resolves via repo-root being on `sys.path`. The engine's
`campaign.boot()` inserts it; the test suite adds it through `pythonpath` in
`graphretr-demo/pyproject.toml`. No install step.

```python
import sys; sys.path.insert(0, "<repo-root>")
from starksearch.graph import RetrievalGraph
from starksearch.qa import Substrate
from starksearch.reward.evaluator import RewardModel
```
