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
        for idx in idxs:
            query, q_id, answer_ids = self._sub.example(idx)
            row = {"idx": int(idx), "q_id": q_id, "query": query,
                   "answer_ids": answer_ids}
            try:
                pred, stats = self._sandbox.run(fn, query, timeout_s=per_query_timeout_s)
                m = self._ev.evaluate(
                    pred, torch.LongTensor(answer_ids), metrics=list(QUALITY_KEYS))
                row["metrics"] = {k: float(m[k]) for k in QUALITY_KEYS}
                row["retrieved"] = [
                    i for i, _ in sorted(pred.items(), key=lambda t: -t[1])[:20]]
                lat_sum += stats.latency_s
                q_sum += stats.queries
            except SandboxError as e:
                crashed += 1
                row["error"] = str(e)
                row["metrics"] = self._zero_quality()
                row["retrieved"] = []
            except Exception as e:
                crashed += 1
                row["error"] = f"{type(e).__name__}: {e}"
                row["metrics"] = self._zero_quality()
                row["retrieved"] = []
            rows.append(row)

        n = max(1, len(rows))
        quality = {k: sum(r["metrics"][k] for r in rows) / n for k in QUALITY_KEYS}
        mv = MetricVector(
            quality=quality,
            latency_s=lat_sum / n,
            db_load=q_sum / n,
            code_complexity=code_complexity(src) if src else 0.0,
            crashed_frac=crashed / n,
        )
        if crashed > self._crash_limit * n:
            mv.quality = self._zero_quality()
            mv.crashed = True
        return (mv, rows) if return_rows else mv
