"""McqRewardAdapter -- make McqReward callable exactly where RewardModel is.

FastLoop._score_candidate (and the rollout / minibatch / meta / final-select call
sites) all call `self._reward.score(fn, idxs, src=..., return_rows=..., \
per_query_timeout_s=...)`. This adapter exposes that SAME signature but holds the
SearchTarget + McqReward + QaSubstrate, so the loop is swapped to score the real
search service with a one-line wiring change (campaign.boot_search) and no edits
to the loop body.

`fn` here is the FileSet (NullSandbox.compile returns it; FileSet.src is the
FileSet itself), and `src` is ignored -- the adapter scores the FileSet it is
handed.

Row translation (the textual-gradient bridge): McqReward emits clean MCQ rows;
FastLoop._reflect buckets on `recall@20`/`hit@1`. We map
  recall@20 <- retrieval_hit  (gold source in context?  -> a RETRIEVAL miss when 0)
  hit@1     <- openbook_correct (right answer from context? -> a REASONING miss
                                  when gold WAS retrieved but answer is wrong)
so _reflect's missed/misranked split becomes exactly the retrieval-vs-reasoning
signal the plan wants -- with NO change to the loop. `retrieved`/`answer_ids` are
left empty so the loop's node-id machinery (get_text) stays inert.
"""
from graphretr_opt.reward.objectives import code_complexity  # noqa: F401  (re-exported convenience)


class McqRewardAdapter:
    def __init__(self, target, mcq_reward, substrate, default_timeout_s=120.0):
        self._target = target
        self._mcq = mcq_reward
        self._sub = substrate
        self._default_timeout_s = float(default_timeout_s)

    def score(self, fn, idxs, src=None, return_rows=False,
              per_query_timeout_s=None):
        """fn: the candidate FileSet. -> MetricVector (and reflect-shaped rows if
        return_rows)."""
        file_set = fn
        timeout = (per_query_timeout_s if per_query_timeout_s is not None
                   else self._default_timeout_s)
        mv, mcq_rows = self._mcq.score(self._target, file_set, idxs, self._sub,
                                       timeout, return_rows=True)
        if not return_rows:
            return mv
        return mv, [self._to_reflect_row(r) for r in mcq_rows]

    @staticmethod
    def _to_reflect_row(r):
        """McqReward row -> FastLoop._reflect / _failure_record contract, with the
        MCQ recap fields carried alongside for the MLflow log_table."""
        rh = 1.0 if r.get("retrieval_hit") else 0.0
        ob = 1.0 if r.get("openbook_correct") else 0.0
        has_ctx = bool((r.get("context_preview") or "")) and not r.get("error")
        return {
            "idx": r["idx"],
            "q_id": r.get("q_id"),
            "query": r.get("question"),
            "answer_ids": [],          # no node ids on this path -> get_text inert
            "metrics": {"recall@20": rh, "hit@1": ob, "hit@5": ob, "mrr": ob},
            "retrieved": [] if not has_ctx else [(0, 1.0)],
            "gold_ranks": {},
            "recall@100": rh,
            "error": r.get("error"),
            # ---- MCQ recap fields (for substrate.eval_log_rows / log_table) ----
            "chosen_idx": r.get("chosen_idx"),
            "gold_idx": r.get("gold_idx"),
            "openbook_correct": bool(r.get("openbook_correct")),
            "closedbook_correct": bool(r.get("closedbook_correct")),
            "retrieval_hit": bool(r.get("retrieval_hit")),
            "context_preview": r.get("context_preview"),
        }
