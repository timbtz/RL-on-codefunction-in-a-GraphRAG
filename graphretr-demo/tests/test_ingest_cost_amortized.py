"""Unit tests for Phase-2 co-optimize: the amortized ingest-cost axis on
MetricVector + Pareto dominance (objectives.py / pareto.py).

The Phase-2 target meters the ONE-TIME USD to build the graph (LLM extraction
during ingest) and amortizes it across the exam:

    total_usd_per_query = usd_cost + ingest_usd / max(1, n_queries)

Selection ranks on (accuracy vs total_usd_per_query), so an expensive richer
graph is accepted only if its accuracy gain, averaged over ALL queries, clears
its amortized build cost -- worth it if it lifts accuracy broadly, wasteful if
it helps one query.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_ingest_cost_amortized
No FalkorDB / Neo4j / network needed.
"""
from graphretr_opt.reward.objectives import MetricVector
from graphretr_opt.reward.pareto import dominates


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _mv(acc, usd=0.0, ingest_usd=0.0, n=35):
    return MetricVector(
        quality={"mcq_accuracy": acc, "retrieval_hit": acc},
        usd_cost=usd, ingest_usd=ingest_usd, n_queries=n,
        primary_key="mcq_accuracy")


def test_amortization_arithmetic():
    # zero-LLM seed: total == usd_cost (backward compatible with graph_search).
    a = _mv(0.5, usd=0.02, ingest_usd=0.0)
    _check("zero-ingest total == usd_cost", abs(a.total_usd_per_query - 0.02) < 1e-12)
    # 0.35 USD ingest over 35 queries -> +0.01/query.
    b = _mv(0.5, usd=0.02, ingest_usd=0.35, n=35)
    _check("amortized total = usd + ingest/n", abs(b.total_usd_per_query - 0.03) < 1e-12)
    # n_queries=0 is treated as 1 (never divide by zero).
    c = _mv(0.5, usd=0.0, ingest_usd=0.5, n=0)
    _check("n=0 amortizes over 1", abs(c.total_usd_per_query - 0.5) < 1e-12)


def test_same_accuracy_cheaper_ingest_dominates():
    # Same accuracy; the candidate that built the graph cheaply dominates the one
    # that spent 100x on ingest for no extra accuracy.
    cheap = _mv(0.55, usd=0.02, ingest_usd=0.35, n=35)   # total 0.03
    pricey = _mv(0.55, usd=0.02, ingest_usd=35.0, n=35)  # total 1.02
    _check("cheap-ingest dominates pricey-ingest (same acc)", dominates(cheap, pricey))
    _check("pricey does NOT dominate cheap", not dominates(pricey, cheap))


def test_amortized_beats_perquery_charge():
    # The SAME 0.35 USD, charged once for the build (amortized) vs charged per
    # query, at equal accuracy: amortized dominates (far lower total cost).
    amortized = _mv(0.55, usd=0.02, ingest_usd=0.35, n=35)  # total 0.03
    perquery = _mv(0.55, usd=0.37, ingest_usd=0.0, n=35)    # total 0.37
    _check("amortized ingest dominates per-query charge", dominates(amortized, perquery))


def test_accuracy_gain_must_clear_amortized_cost():
    # Cost-weighted composite (the gate's blend uses mv.get(); total_usd_per_query
    # is reachable via get()). A richer, more-expensive graph is admitted only if
    # its accuracy gain clears the amortized cost at the chosen weight.
    def composite(mv, w):
        return mv.get("mcq_accuracy") - w * mv.get("total_usd_per_query")

    base = _mv(0.50, usd=0.02, ingest_usd=0.0)            # total 0.02
    rich = _mv(0.60, usd=0.02, ingest_usd=1.75, n=35)     # +0.05/q -> total 0.07
    # delta_acc = 0.10, delta_cost = 0.05. Crossover weight = 0.10/0.05 = 2.0.
    _check("rich wins when gain clears cost (w=1)", composite(rich, 1.0) > composite(base, 1.0))
    _check("rich loses when cost weighted high (w=3)", composite(rich, 3.0) < composite(base, 3.0))
    # total_usd_per_query is reachable through the cost-aware get() fallback.
    _check("get(total_usd_per_query) works", abs(rich.get("total_usd_per_query") - 0.07) < 1e-12)


def test_dominance_frontier_tradeoff():
    # Higher accuracy + higher total cost => neither dominates (both on frontier):
    # the amortized cost does not let a cheaper-but-worse candidate dominate a
    # pricier-but-better one, and vice versa.
    better = _mv(0.60, usd=0.02, ingest_usd=1.75, n=35)  # acc 0.60 / total 0.07
    cheaper = _mv(0.50, usd=0.02, ingest_usd=0.0)        # acc 0.50 / total 0.02
    _check("better does NOT dominate cheaper (costs more)", not dominates(better, cheaper))
    _check("cheaper does NOT dominate better (less accurate)", not dominates(cheaper, better))


def main():
    test_amortization_arithmetic()
    test_same_accuracy_cheaper_ingest_dominates()
    test_amortized_beats_perquery_charge()
    test_accuracy_gain_must_clear_amortized_cost()
    test_dominance_frontier_tradeoff()
    print("\nall amortized-cost tests passed")


if __name__ == "__main__":
    main()
