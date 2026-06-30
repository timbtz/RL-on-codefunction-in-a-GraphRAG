"""StarkRewardAdapter -- make the subprocess STaRK target callable exactly where
the in-process RewardModel was.

FastLoop (and the rollout / minibatch / meta / final-select call sites) all call
`self._reward.score(fn, idxs, src=..., return_rows=..., per_query_timeout_s=...)`.
This adapter exposes that SAME signature but holds a StarkSubprocessTarget +
Substrate: it batches the gate queries through ONE worker subprocess, scores each
returned pred with the STaRK Evaluator, and builds the identical MetricVector +
reflection rows the in-process RewardModel produced. `fn` is ignored (the
NullSandbox hands the loop the program SOURCE, passed as `src`) -- the subprocess
is the isolation, replacing the deleted in-process Sandbox/AST gate.

The scoring/row logic is byte-for-byte the old RewardModel's, with the per-query
`Sandbox.run(fn, query) -> (pred, stats)` swapped for a batched
`target.run(src, queries) -> {query: {pred, cost, error}}`.
"""
import torch

from graphretr_opt.reward.objectives import (
    MetricVector, code_complexity, code_tokens)
from .evaluator import QUALITY_KEYS


def _primary_src(src) -> str:
    """The candidate's PRIMARY editable-file source as a string. `src` is a
    FileSet (the editable service tree); a plain string (legacy / tests) flows
    straight through. -> str ('' when src is None)."""
    if src is None:
        return ""
    if hasattr(src, "overlay"):                       # FileSet
        return src.src_of(src.primary_file)
    return src


def _src_complexity(src) -> float:
    """AST-complexity proxy (Pareto diagnostic axis)."""
    s = _primary_src(src)
    return code_complexity(s) if s else 0.0


def _src_tokens(src) -> float:
    """Program SIZE in source tokens -- the value gate's "K" complexity axis."""
    s = _primary_src(src)
    return code_tokens(s) if s else 0.0


class StarkRewardAdapter:
    def __init__(self, substrate, target, crash_frac_limit=0.10,
                 default_timeout_s=30.0):
        self._sub = substrate
        self._target = target
        self._ev = substrate.make_evaluator()
        self._crash_limit = crash_frac_limit
        self._default_timeout_s = float(default_timeout_s)

    def _zero_quality(self):
        return {k: 0.0 for k in QUALITY_KEYS}

    def score(self, fn, idxs, src=None, return_rows=False, per_query_timeout_s=None):
        """src: the candidate program source (required -- the subprocess runs it).
        -> MetricVector (and per-query rows if return_rows)."""
        if src is None:
            raise ValueError("StarkRewardAdapter.score needs src (the candidate "
                             "program source); fn is ignored on the subprocess path")
        idxs = list(idxs)
        # (idx, query, q_id, answer_ids) -- several idxs may share a query string,
        # so we key the worker results by query and re-attach per idx below.
        examples = [(int(i), *self._sub.example(i)) for i in idxs]
        queries = [q for _, q, _, _ in examples]
        per_q = (per_query_timeout_s if per_query_timeout_s is not None
                 else self._default_timeout_s)
        # One subprocess runs the whole batch; the wall-clock kill is generous
        # (per-query budget * n) -- a kill switch, not a pacing delay.
        batch_timeout = max(per_q, per_q * len(set(queries)))
        results = self._target.run(src, queries, batch_timeout)

        rows = []
        crashed = 0
        lat_sum = q_sum = llm_sum = rerank_sum = 0.0
        usd_sum = tin_sum = tout_sum = 0.0
        for idx, query, q_id, answer_ids in examples:
            r = results.get(query) or {"pred": {}, "cost": {}, "error": "no result"}
            row = {"idx": idx, "q_id": q_id, "query": query,
                   "answer_ids": answer_ids}
            cost = r.get("cost") or {}
            lat_sum += float(cost.get("latency_s", 0.0))
            q_sum += float(cost.get("db_queries", 0))
            llm_sum += float(cost.get("llm_calls", 0))
            rerank_sum += float(cost.get("rerank_items", 0))
            # The CostSink already meters real $ + tokens per query (the worker
            # snapshots it into `cost`); the old adapter dropped them on the floor.
            # Sum them so usd_cost lands on the vector -> Pareto axis + value gate.
            usd_sum += float(cost.get("usd_cost", 0.0))
            tin_sum += float(cost.get("tokens_in", 0))
            tout_sum += float(cost.get("tokens_out", 0))
            pred = r.get("pred") or {}
            if r.get("error") or not pred:
                crashed += 1
                row["error"] = r.get("error") or "search() returned an empty dict"
                row["metrics"] = self._zero_quality()
                row["retrieved"] = []
                row["gold_ranks"] = {}
                row["recall@100"] = 0.0
                rows.append(row)
                continue
            m = self._ev.evaluate(
                pred, torch.LongTensor(answer_ids), metrics=list(QUALITY_KEYS))
            row["metrics"] = {k: float(m[k]) for k in QUALITY_KEYS}
            ranked = sorted(pred.items(), key=lambda t: -t[1])
            row["retrieved"] = [(int(i), float(s)) for i, s in ranked[:20]]
            pos = {int(i): rk for rk, (i, _) in enumerate(ranked)}
            row["gold_ranks"] = {a: (pos[a], float(pred[a]))
                                 for a in answer_ids if a in pos}
            gold = set(answer_ids)
            top100 = {int(i) for i, _ in ranked[:100]}
            row["recall@100"] = (len(gold & top100) / len(gold)) if gold else 0.0
            rows.append(row)

        n = max(1, len(rows))
        quality = {k: sum(r["metrics"][k] for r in rows) / n for k in QUALITY_KEYS}
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
            code_complexity=_src_complexity(src),
            code_tokens=_src_tokens(src),
            usd_cost=usd_sum / n,
            tokens_in=tin_sum / n,
            tokens_out=tout_sum / n,
            crashed_frac=crashed / n,
            recall_at_100=sum(r.get("recall@100", 0.0) for r in rows) / n,
            per_query=per_query,
        )
        if crashed > self._crash_limit * n:
            mv.quality = self._zero_quality()
            mv.crashed = True
        mv.sanitize()
        return (mv, rows) if return_rows else mv
