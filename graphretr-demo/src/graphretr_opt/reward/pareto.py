"""pareto.py -- the multi-objective dominance test.

Objectives (all "higher is better" after sign-flipping costs):
  + quality recall@20   (maximize)
  - latency_s           (minimize -> negate)
  - db_load             (minimize -> negate)
  - code_complexity     (minimize -> negate)
  - total_usd_per_query (minimize -> negate) -- amortized TOTAL-pipeline $/query:
                        per-query search USD + one-time ingest USD / exam size.
                        Equals usd_cost whenever ingest is zero-LLM (function /
                        STaRK paths, and the seed ingest_search candidate), so this
                        is backward-compatible; it only diverges once the Phase-2
                        co-optimize target meters LLM-extraction during ingest.

`dominates` is the live primitive the CandidatePool (pool.py) and the gate
(gate.py) build on. The passive `ParetoArchive`/MAP-Elites record that once sat
beside it was load-bearing for nothing (its `.entries`/`.cells`/`.best()` were
never read outside the class) and has been removed; the instance-wise pool is
the active frontier now.
"""


def _objective_tuple(mv):
    # total_usd_per_query falls back to usd_cost when ingest_usd == 0, so existing
    # targets are unaffected; the ingest_search target gets amortized total cost.
    return (mv.primary, -mv.latency_s, -mv.db_load, -mv.code_complexity,
            -mv.total_usd_per_query)


def dominates(a, b) -> bool:
    """a dominates b: >= on every objective and > on at least one."""
    ta, tb = _objective_tuple(a), _objective_tuple(b)
    return all(x >= y for x, y in zip(ta, tb)) and any(x > y for x, y in zip(ta, tb))
