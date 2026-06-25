"""RewardModel -- run a program over a set of queries in the sandbox and score
it into a MetricVector (quality via STaRK Evaluator node-containment, plus the
latency / db-load / complexity probes).

score() also optionally returns per-query rows -- that is what the fast loop's
rollout uses to harvest the worst misses for the reflection prompt (rows are
diagnostics, never the gate).
"""
import torch

from ..env.sandbox import SandboxError
from .objectives import MetricVector, QUALITY_KEYS, code_complexity


class RewardModel:
    def __init__(self, substrate, sandbox, crash_frac_limit=0.10):
        self._sub = substrate
        self._sandbox = sandbox
        self._ev = substrate.make_evaluator()
        self._crash_limit = crash_frac_limit

    def _zero_quality(self):
        return {k: 0.0 for k in QUALITY_KEYS}

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        """fn: compiled candidate; idxs: positional query indices.
        -> MetricVector (and rows if return_rows)."""
        rows = []
        crashed = 0
        lat_sum = 0.0
        q_sum = 0
        llm_sum = 0
        rerank_sum = 0
        for idx in idxs:
            query, q_id, answer_ids = self._sub.example(idx)
            row = {"idx": int(idx), "q_id": q_id, "query": query,
                   "answer_ids": answer_ids}
            try:
                pred, stats = self._sandbox.run(fn, query, timeout_s=per_query_timeout_s)
                m = self._ev.evaluate(
                    pred, torch.LongTensor(answer_ids), metrics=list(QUALITY_KEYS))
                row["metrics"] = {k: float(m[k]) for k in QUALITY_KEYS}
                # Keep scores + gold ranks so the reflection prompt can show WHAT
                # out-ranked the gold (ranking-failure diagnosis, not just recall).
                ranked = sorted(pred.items(), key=lambda t: -t[1])
                row["retrieved"] = [(int(i), float(s)) for i, s in ranked[:20]]
                pos = {int(i): r for r, (i, _) in enumerate(ranked)}
                row["gold_ranks"] = {a: (pos[a], float(pred[a]))
                                     for a in answer_ids if a in pos}
                # recall@100 (Phase C2): gold present ANYWHERE in the program's
                # top-100 -- "reachability into the final dict", not an abstract
                # pool. recall@100 >> recall@20 => gold generated but mis-ranked
                # (fix ranking); recall@100 low => gold not generated (fix
                # generation: reformulate/expand). NON-gated diagnostic.
                gold = set(answer_ids)
                top100 = {int(i) for i, _ in ranked[:100]}
                row["recall@100"] = (len(gold & top100) / len(gold)) if gold else 0.0
                lat_sum += stats.latency_s
                q_sum += stats.queries
                llm_sum += stats.llm_calls
                rerank_sum += stats.rerank_items
            except SandboxError as e:
                crashed += 1
                row["error"] = str(e)
                row["metrics"] = self._zero_quality()
                row["retrieved"] = []
                row["gold_ranks"] = {}
                row["recall@100"] = 0.0
            except Exception as e:
                crashed += 1
                row["error"] = f"{type(e).__name__}: {e}"
                row["metrics"] = self._zero_quality()
                row["retrieved"] = []
                row["gold_ranks"] = {}
                row["recall@100"] = 0.0
            rows.append(row)

        n = max(1, len(rows))
        quality = {k: sum(r["metrics"][k] for r in rows) / n for k in QUALITY_KEYS}
        # Per-query retention (Phase A1): keep the instance-wise signals the
        # candidate pool selects on -- reciprocal rank (mrr), hit@1, and the
        # recall@100 reachability flag. Cheap (~one small dict per gate query).
        per_query = {int(r["idx"]): {"mrr": r["metrics"]["mrr"],
                                     "hit@1": r["metrics"]["hit@1"],
                                     "recall@100": r.get("recall@100", 0.0)}
                     for r in rows}
        mv = MetricVector(
            quality=quality,
            latency_s=lat_sum / n,
            db_load=q_sum / n,
            llm_calls=llm_sum / n,
            rerank_items=rerank_sum / n,
            code_complexity=code_complexity(src) if src else 0.0,
            crashed_frac=crashed / n,
            recall_at_100=sum(r.get("recall@100", 0.0) for r in rows) / n,
            per_query=per_query,
        )
        if crashed > self._crash_limit * n:
            mv.quality = self._zero_quality()
            mv.crashed = True
        # Phase 2 vector-wise guard: a candidate whose math yields a NaN/inf on any
        # axis degrades to a safe-worst (crashed) vector instead of poisoning the
        # pool's dominance test / the gate comparison. Per-axis, never scalarized.
        mv.sanitize()
        return (mv, rows) if return_rows else mv
