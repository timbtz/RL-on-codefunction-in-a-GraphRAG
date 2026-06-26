"""pareto.py -- the multi-objective dominance test.

Objectives (all "higher is better" after sign-flipping costs):
  + quality recall@20   (maximize)
  - latency_s           (minimize -> negate)
  - db_load             (minimize -> negate)
  - code_complexity     (minimize -> negate)
  - usd_cost            (minimize -> negate) -- real $/query (graph_search path;
                        0 and inert on the function path, where no LLM is metered)

`dominates` is the live primitive the CandidatePool (pool.py) and the gate
(gate.py) build on. The passive `ParetoArchive`/MAP-Elites record that once sat
beside it was load-bearing for nothing (its `.entries`/`.cells`/`.best()` were
never read outside the class) and has been removed; the instance-wise pool is
the active frontier now.
"""


def _objective_tuple(mv):
    return (mv.primary, -mv.latency_s, -mv.db_load, -mv.code_complexity,
            -mv.usd_cost)


def dominates(a, b) -> bool:
    """a dominates b: >= on every objective and > on at least one."""
    ta, tb = _objective_tuple(a), _objective_tuple(b)
    return all(x >= y for x, y in zip(ta, tb)) and any(x > y for x, y in zip(ta, tb))
