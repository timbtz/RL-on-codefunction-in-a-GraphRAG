"""Unit tests for the MCQ reward stack (reward/mcq_reward.py, qa_objectives.py)
with FakeSearchTarget + a mock answerer -- fully deterministic, no judge to stub,
no Neo4j, no API keys.

Covers: exact-match scoring, closed-book subtraction (openbook - closedbook),
retrieval-hit, crash-frac zeroing, MetricVector shape/primary_key, recap rows,
unparseable-answer fallback, and the eval-hygiene invariant (choices/answer_idx
never cross the SearchTarget seam).
"""
from graphretr_opt.artifact.file_set import FileSet
from graphretr_opt.env.search_target import CostMeter, SearchResult
from graphretr_opt.env.targets.fake_target import FakeSearchTarget
from graphsearch.reward.mcq_reward import McqReward
from graphsearch.reward.qa_objectives import (MCQ_PRIMARY, mcq_dominates,
                                                mcq_objective_tuple)

FS = FileSet("/b", {"a.py": "x = 1\n"}, ("a.py",))


class MockAnswerer:
    """Returns a configured letter per question text."""
    def __init__(self, by_q):
        self.by_q = by_q

    def invoke(self, messages):
        user = next(m["content"] for m in messages if m["role"] == "user")
        q = user.split("Question:\n", 1)[1].split("\n\nOptions:", 1)[0]
        class _M:
            content = self_letter = None
        m = _M()
        m.content = self.by_q.get(q, "A")
        return m


class FakeSubstrate:
    def __init__(self, items):
        self._items = items  # list of (question, q_id, gold)

    def example(self, idx):
        return self._items[int(idx)]


def _gold(answer_idx, closedbook, source="src0", n=4):
    return {"choices": [f"opt{i}" for i in range(n)] + ["cannot"],
            "answer_idx": answer_idx, "source": source,
            "closedbook_correct": closedbook}


def test_exact_match_and_closedbook_subtraction():
    items = [
        ("Q0?", "q0", _gold(0, closedbook=False)),   # answerer right, cb wrong -> +1
        ("Q1?", "q1", _gold(0, closedbook=True)),    # answerer right, cb right -> 0
    ]
    sub = FakeSubstrate(items)
    target = FakeSearchTarget(contexts={"Q0?": "ctx0", "Q1?": "ctx1"})
    r = McqReward(MockAnswerer({"Q0?": "A", "Q1?": "A"}))
    mv = r.score(target, FS, [0, 1], sub, timeout_s=5)
    assert abs(mv.quality["mcq_accuracy"] - 0.5) < 1e-9      # (1 + 0) / 2
    assert abs(mv.quality["openbook_accuracy"] - 1.0) < 1e-9
    assert mv.primary_key == MCQ_PRIMARY
    assert abs(mv.primary - 0.5) < 1e-9


def test_retrieval_hit():
    items = [("Q0?", "q0", _gold(0, False, source="ACME")),
             ("Q1?", "q1", _gold(0, False, source="ZZZ"))]
    sub = FakeSubstrate(items)
    target = FakeSearchTarget(contexts={"Q0?": "see ACME report", "Q1?": "nope"})
    mv = r = McqReward(MockAnswerer({})).score(
        target, FS, [0, 1], sub, timeout_s=5)
    assert abs(mv.quality["retrieval_hit"] - 0.5) < 1e-9


def test_crash_frac_zeroes_quality():
    class CrashTarget:
        def run(self, fs, queries, timeout_s):
            return {q: SearchResult(context="", cost=CostMeter(), error="boom")
                    for q in queries}
    items = [("Q0?", "q0", _gold(0, False)), ("Q1?", "q1", _gold(0, False))]
    mv = McqReward(MockAnswerer({})).score(
        CrashTarget(), FS, [0, 1], FakeSubstrate(items), timeout_s=5)
    assert mv.crashed is True
    assert mv.quality["mcq_accuracy"] == 0.0


def test_unparseable_answer_falls_back_to_cannot_determine():
    items = [("Q0?", "q0", _gold(0, False))]
    sub = FakeSubstrate(items)
    target = FakeSearchTarget(contexts={"Q0?": "ctx"})
    mv, rows = McqReward(MockAnswerer({"Q0?": "banana"})).score(
        target, FS, [0], sub, timeout_s=5, return_rows=True)
    # 'banana' -> no valid leading letter in range -> cannot-determine (last idx)
    assert rows[0]["chosen_idx"] == len(items[0][2]["choices"]) - 1
    assert rows[0]["openbook_correct"] is False


def test_recap_rows_shape():
    items = [("Q0?", "q0", _gold(1, False))]
    sub = FakeSubstrate(items)
    target = FakeSearchTarget(contexts={"Q0?": "ctx"})
    mv, rows = McqReward(MockAnswerer({"Q0?": "B"})).score(
        target, FS, [0], sub, timeout_s=5, return_rows=True)
    row = rows[0]
    for k in ("q_id", "chosen_idx", "gold_idx", "openbook_correct",
              "closedbook_correct", "retrieval_hit", "context_preview"):
        assert k in row
    assert row["chosen_idx"] == 1 and row["gold_idx"] == 1
    assert row["openbook_correct"] is True


def test_eval_hygiene_only_question_crosses_seam():
    seen = {}

    class SpyTarget:
        def run(self, fs, queries, timeout_s):
            seen["queries"] = list(queries)
            return {q: SearchResult(context="ctx", cost=CostMeter()) for q in queries}

    items = [("Q0?", "q0", _gold(2, False, n=4))]
    McqReward(MockAnswerer({})).score(SpyTarget(), FS, [0],
                                      FakeSubstrate(items), timeout_s=5)
    # the target only ever saw the question -- not the choices or the answer key
    assert seen["queries"] == ["Q0?"]
    blob = " ".join(seen["queries"])
    assert "cannot" not in blob and "opt2" not in blob


def test_mcq_dominance_tuple():
    from graphretr_opt.reward.objectives import MetricVector
    hi = MetricVector(quality={"mcq_accuracy": 0.8, "retrieval_hit": 0.5},
                      latency_s=1.0, llm_calls=3, primary_key=MCQ_PRIMARY)
    lo = MetricVector(quality={"mcq_accuracy": 0.5, "retrieval_hit": 0.5},
                      latency_s=1.0, llm_calls=3, primary_key=MCQ_PRIMARY)
    assert mcq_dominates(hi, lo)
    assert not mcq_dominates(lo, hi)
    assert mcq_objective_tuple(hi)[0] == 0.8
