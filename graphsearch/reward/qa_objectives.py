"""qa_objectives.py -- the MCQ multi-objective keys for the graph_search target.

Reuses MetricVector (objectives.py) unchanged -- the Pareto/pool/gate/checkpoint
code stays artifact-agnostic. The only difference is the quality dict's KEYS and
which one is `primary`:

  quality = {"mcq_accuracy": float, "retrieval_hit": float}

`mcq_accuracy` is the CLOSED-BOOK-ADJUSTED exact-match rate
(mean(openbook_correct - closedbook_correct)), so only the *retrieval
contribution* is rewarded (Risk R6). Raw `openbook_accuracy` and the
`closedbook_delta` are logged-but-not-dominating (kept in the recap rows / flat
metrics, not in the dominance tuple).

The active dominance ranks on `primary` (= mcq_accuracy when the reward sets
primary_key) plus the shared cost axes via reward.pareto. `mcq_objective_tuple`
is the explicit MCQ dominance order (with retrieval_hit as a co-objective),
exposed for tests and any future MCQ-specific Pareto pool.
"""

MCQ_QUALITY_KEYS = ("mcq_accuracy", "retrieval_hit")
MCQ_PRIMARY = "mcq_accuracy"


def mcq_objective_tuple(mv):
    """MCQ dominance order: maximize accuracy then retrieval-hit, minimize cost.
    Cost is REAL USD (OpenRouter token_usage.cost), NOT raw call count -- one
    big-prompt call can cost more than several tiny ones, so dollars is the right
    unit. latency is the secondary cost axis. All "higher is better" after
    sign-flipping costs."""
    return (mv.get("mcq_accuracy"), mv.get("retrieval_hit"),
            -mv.total_usd_per_query, -mv.latency_s)


def mcq_dominates(a, b) -> bool:
    """a dominates b on the MCQ objective tuple (>= on all, > on at least one)."""
    ta, tb = mcq_objective_tuple(a), mcq_objective_tuple(b)
    return (all(x >= y for x, y in zip(ta, tb))
            and any(x > y for x, y in zip(ta, tb)))
