"""McqReward -- score a candidate FileSet by RUNNING the real search service and
turning the retrieved context into a deterministic multiple-choice answer.

NO LLM judge. The only quality call per question is the answerer's single-letter
selection, scored by EXACT-MATCH against the gold answer_idx -- so the reward is
deterministic (temperature=0 answerer) and judge-free. Retrieval quality is
isolated by a CLOSED-BOOK baseline subtraction:

    mcq_accuracy = mean(openbook_correct - closedbook_correct)

where closedbook_correct is precomputed offline (mcq_gen, candidate-independent)
and read straight off the gold item -- never recomputed in the loop. A cheap,
judge-free retrieval_hit (does the gold source appear in the returned context) is
a secondary axis.

Eval hygiene: only the QUESTION crosses the SearchTarget seam. The choices and
the answer_idx stay here; the search service never sees the options or the key.

Sibling to reward/evaluator.py:RewardModel (which scores the function campaign
via the Sandbox). Same MetricVector out, so Pareto/pool/gate/checkpoint are
reused unchanged.
"""
import re
import string
import time

from graphretr_opt.reward.objectives import MetricVector, code_complexity
from .qa_objectives import MCQ_PRIMARY, MCQ_QUALITY_KEYS

# Mirror of graphsearch/mcq_gen/mcq_format (kept in sync by hand) -- the hot loop
# must not depend on graphsearch being importable.
CANNOT_DETERMINE_HINT = ("If the context does not contain enough information, "
                         "choose the option that says it cannot be determined.")
ANSWER_SYSTEM = (
    "You answer a multiple-choice question using ONLY the provided context. "
    "Choose the single best option. " + CANNOT_DETERMINE_HINT + " "
    "Reply with EXACTLY one capital letter (A, B, C, ...) and nothing else."
)


def _render_choices(choices):
    return "\n".join(f"{l}) {c}"
                     for l, c in zip(string.ascii_uppercase, choices))


def _parse_letter(text, n):
    """Strict single-letter parse -> 0-based idx, or None (unparseable).
    Accepts a STANDALONE letter only ("A", "A)", "answer: C.") -- a word like
    "banana" has no single-letter token and parses to None (-> cannot-determine)."""
    if not text:
        return None
    m = re.search(r"\b([A-Za-z])\b", text)
    if not m:
        return None
    idx = string.ascii_uppercase.index(m.group(1).upper())
    return idx if 0 <= idx < n else None


class _StubMsg:
    def __init__(self, content):
        self.content = content


class StubAnswerer:
    """A deterministic, API-key-free answerer for offline/fake-target smoke runs
    (Task 15). It is NOT a real model: it scores each choice by word-overlap with
    the retrieved context and returns that letter, falling back to the
    cannot-determine option when the context is empty. Enough to exercise the full
    loop (mutate -> score -> Pareto -> checkpoint -> recap table) with no infra."""

    def invoke(self, messages):
        user = ""
        for m in messages:
            if (m.get("role") if isinstance(m, dict) else None) == "user":
                user = m.get("content", "")
        context, options = self._split(user)
        if not context.strip() or context.strip() == "(no context retrieved)":
            return _StubMsg(string.ascii_uppercase[max(0, len(options) - 1)])
        ctx_words = set(re.findall(r"\w+", context.lower()))
        best_i, best_score = 0, -1
        for i, opt in enumerate(options):
            ow = set(re.findall(r"\w+", opt.lower()))
            score = len(ctx_words & ow)
            if score > best_score:
                best_i, best_score = i, score
        return _StubMsg(string.ascii_uppercase[best_i])

    @staticmethod
    def _split(user):
        """Pull the context block and the lettered options back out of the prompt
        rendered by McqReward._answer."""
        context = ""
        if "Context:\n" in user and "\n\nQuestion:" in user:
            context = user.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        options = []
        if "Options:\n" in user:
            opt_block = user.split("Options:\n", 1)[1].split("\n\nAnswer", 1)[0]
            for line in opt_block.splitlines():
                m = re.match(r"\s*[A-Z]\)\s*(.*)", line)
                if m:
                    options.append(m.group(1))
        return context, options


def _answerer_cost(res) -> dict:
    """REAL USD + tokens for one answerer call. OpenRouter returns the dollar
    cost in response_metadata.token_usage.cost (usage accounting enabled on the
    answerer in campaign._build_answerer). StubAnswerer / non-OpenRouter models
    carry no such metadata -> zeros."""
    meta = getattr(res, "response_metadata", None) or {}
    usage = getattr(res, "usage_metadata", None) or {}
    try:
        usd = float((meta.get("token_usage") or {}).get("cost") or 0.0)
    except (TypeError, ValueError):
        usd = 0.0
    try:
        tin = int(usage.get("input_tokens", 0) or 0)
        tout = int(usage.get("output_tokens", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        tin = tout = 0
    return {"usd": usd, "tokens_in": tin, "tokens_out": tout}


class McqReward:
    def __init__(self, answerer_llm, crash_frac_limit=0.10, answer_concurrency=8):
        self._answerer = answerer_llm          # injected BaseChatModel (mock in tests)
        self._crash_limit = crash_frac_limit
        # The answerer calls are independent (one MCQ each) and read cost off
        # their own response -> no shared sink -> safe to fan out. Light calls, so
        # cap modestly; the OpenRouter key handles >>this (see graphsearch/.env).
        self._answer_concurrency = max(1, int(answer_concurrency or 1))

    # ------------------------------------------------------------------ #
    def _answer(self, context, question, choices):
        """One constrained MCQ selection. Returns (chosen_idx, raw_text, cost).
        `cost` is {usd, tokens_in, tokens_out} for THIS answerer call (the
        answerer's input grows with retrieved context, so it is candidate-
        dependent cost). On an unparseable reply, falls back to the
        cannot-determine option (last)."""
        user = (f"Context:\n{context or '(no context retrieved)'}\n\n"
                f"Question:\n{question}\n\nOptions:\n{_render_choices(choices)}\n\n"
                "Answer with one letter:")
        res = self._answerer.invoke([
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user},
        ])
        raw = getattr(res, "content", res)
        raw = raw if isinstance(raw, str) else str(raw)
        chosen = _parse_letter(raw, len(choices))
        if chosen is None:
            chosen = len(choices) - 1          # cannot-determine fallback
        return chosen, raw.strip(), _answerer_cost(res)

    @staticmethod
    def _retrieval_hit(source, context):
        """Cheap, judge-free: do the gold source(s) appear in the returned
        context? Substring match (case-insensitive). Secondary axis.

        `source` is either a single gold node id (str) OR a list/tuple of ids
        for a MULTI-EVIDENCE question -- then ALL of them must be retrieved for
        a hit (a set-answer needs the whole supporting set in context, not one
        node). Backward compatible: a plain string behaves exactly as before."""
        if not source or not context:
            return False
        ctx = context.lower()
        ids = [source] if isinstance(source, str) else list(source)
        ids = [str(s).lower() for s in ids if s]
        return bool(ids) and all(s in ctx for s in ids)

    def score(self, target, file_set, idxs, substrate, timeout_s,
              return_rows=False):
        """Run `target` over the questions at `idxs`, answer each MCQ, and
        aggregate a MetricVector (+ per-question recap rows). Only the question
        strings cross into `target`."""
        idxs = list(idxs)
        examples = [substrate.example(i) for i in idxs]
        queries = [q for q, _, _ in examples]
        results = target.run(file_set, queries, timeout_s)

        # Answer each MCQ. The answerer calls are independent (no shared state),
        # so fan them out -- preserving input order -- and accumulate after.
        def _answer_one(pair):
            (question, q_id, gold), idx = pair
            res = results.get(question)
            error = getattr(res, "error", None) if res is not None else "no result"
            context = getattr(res, "context", "") if res is not None else ""
            cost = getattr(res, "cost", None)
            row = {"idx": int(idx), "q_id": q_id, "question": question,
                   "gold_idx": gold["answer_idx"],
                   "closedbook_correct": bool(gold["closedbook_correct"]),
                   "error": error}
            t_ans = time.time()
            ans_cost = {"usd": 0.0, "tokens_in": 0, "tokens_out": 0}
            is_crash = False
            if error or not context:
                # crash / empty context -> answerer not even called; scored a miss
                if error:
                    is_crash = True
                chosen = len(gold["choices"]) - 1   # cannot-determine
                openbook = False
            else:
                chosen, _raw, ans_cost = self._answer(context, question, gold["choices"])
                openbook = (chosen == gold["answer_idx"])
            ans_secs = time.time() - t_ans
            row["chosen_idx"] = chosen
            row["openbook_correct"] = bool(openbook)
            row["retrieval_hit"] = self._retrieval_hit(gold.get("source"), context)
            row["context_preview"] = (context or "")[:300]
            return row, cost, ans_cost, ans_secs, is_crash

        pairs = list(zip(examples, idxs))
        ac = max(1, min(len(pairs), self._answer_concurrency))
        if ac <= 1:
            processed = [_answer_one(p) for p in pairs]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=ac) as ex:
                processed = list(ex.map(_answer_one, pairs))   # map preserves order

        rows = []
        crashed = 0
        lat_sum = llm_sum = db_sum = 0.0
        usd_sum = tin_sum = tout_sum = 0.0
        for row, cost, ans_cost, ans_secs, is_crash in processed:
            if is_crash:
                crashed += 1
            rows.append(row)
            if cost is not None:
                lat_sum += cost.latency_s + ans_secs
                # +1 llm call for the answerer's selection
                llm_sum += cost.llm_calls + 1
                db_sum += cost.db_queries
                # REAL USD: search calls (metered in the worker) + the answerer
                usd_sum += getattr(cost, "usd_cost", 0.0) + ans_cost["usd"]
                tin_sum += getattr(cost, "tokens_in", 0) + ans_cost["tokens_in"]
                tout_sum += getattr(cost, "tokens_out", 0) + ans_cost["tokens_out"]
            else:
                lat_sum += ans_secs
                llm_sum += 1
                usd_sum += ans_cost["usd"]
                tin_sum += ans_cost["tokens_in"]
                tout_sum += ans_cost["tokens_out"]

        n = max(1, len(rows))
        # closed-book-adjusted accuracy: reward only the retrieval contribution
        adj = [(1.0 if r["openbook_correct"] else 0.0)
               - (1.0 if r["closedbook_correct"] else 0.0) for r in rows]
        openbook_rate = sum(1.0 for r in rows if r["openbook_correct"]) / n
        closedbook_rate = sum(1.0 for r in rows if r["closedbook_correct"]) / n
        quality = {
            "mcq_accuracy": sum(adj) / n,
            "retrieval_hit": sum(1.0 for r in rows if r["retrieval_hit"]) / n,
        }
        # per-question retention (parity with RewardModel.per_query) for any
        # instance-wise pool selection.
        per_query = {int(r["idx"]): {
            "mcq_accuracy": (1.0 if r["openbook_correct"] else 0.0)
                            - (1.0 if r["closedbook_correct"] else 0.0),
            "retrieval_hit": 1.0 if r["retrieval_hit"] else 0.0,
        } for r in rows}

        cc = 0.0
        if file_set is not None:
            cc = sum(code_complexity(s) for s in
                     getattr(file_set, "overlay", {}).values())

        mv = MetricVector(
            quality=quality,
            latency_s=lat_sum / n,
            db_load=db_sum / n,
            llm_calls=llm_sum / n,
            usd_cost=usd_sum / n,
            tokens_in=tin_sum / n,
            tokens_out=tout_sum / n,
            code_complexity=cc,
            crashed_frac=crashed / n,
            per_query=per_query,
            primary_key=MCQ_PRIMARY,
        )
        # log-but-not-dominate diagnostics live as extra quality keys so they ride
        # along into as_flat()/MLflow without entering the dominance tuple.
        mv.quality["openbook_accuracy"] = openbook_rate
        mv.quality["closedbook_rate"] = closedbook_rate
        if crashed > self._crash_limit * n:
            mv.quality = {k: 0.0 for k in MCQ_QUALITY_KEYS}
            mv.crashed = True
        mv.sanitize()
        return (mv, rows) if return_rows else mv
