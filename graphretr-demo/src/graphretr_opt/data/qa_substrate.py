"""QaSubstrate -- the data port for the graph_search target.

Same shape as data/substrate.py:Substrate (example / train_idxs / val_idxs /
gate_idxs / meta_holdout_idxs / rotating_gate_idxs / make_evaluator), but over
the offline MCQ gold set (graphsearch/data/dataset.json) instead of STaRK-prime.
No torch, no stark_qa -- this is what lets the graph_search campaign run without
the function-campaign's dataset deps.

`example(idx)` returns the QUESTION plus a `gold` dict; the question is the ONLY
field that ever crosses the SearchTarget seam. `gold` (choices / answer_idx /
source / closedbook_correct) stays in the optimizer process for the reward --
the search service never sees the options or the key (eval hygiene).
"""
import json
import os
import random


class QaSubstrate:
    def __init__(self, dataset_path="graphsearch/data/dataset.json",
                 meta_holdout_size=0, meta_seed=1234):
        self._path = dataset_path
        with open(dataset_path, encoding="utf-8") as f:
            self._items = json.load(f)
        n = len(self._items)
        # tiny set: train == val == the whole set (the MCQ set is small; see
        # Risk R1). Splits stay positional ints to match Substrate exactly.
        self._train = list(range(n))
        self._val = list(range(n))
        self._meta_holdout, self._gate_pool = self._partition(meta_holdout_size,
                                                              meta_seed)
        print(f"[qa_substrate] loaded {n} MCQ items from {dataset_path} "
              f"(gate-pool {len(self._gate_pool)} / meta-holdout "
              f"{len(self._meta_holdout)})")

    def _partition(self, size, seed):
        """Disjoint (meta_holdout, gate_pool) over val. size<=0 => empty holdout
        (whole val is the gate pool)."""
        if size <= 0:
            return [], list(self._val)
        size = min(int(size), len(self._val))
        holdout = set(random.Random(seed * 7919 + 1).sample(self._val, size))
        return sorted(holdout), [i for i in self._val if i not in holdout]

    @property
    def train_idxs(self):
        return list(self._train)

    @property
    def val_idxs(self):
        return list(self._val)

    @property
    def meta_holdout_idxs(self):
        return list(self._meta_holdout)

    def example(self, idx):
        """Positional idx -> (question, q_id, gold). gold carries the scoring
        material the reward needs (NEVER passed to the SearchTarget)."""
        it = self._items[int(idx)]
        gold = {
            "choices": list(it["choices"]),
            "answer_idx": int(it["answer_idx"]),
            "source": it.get("source", ""),
            "closedbook_correct": bool(it.get("closedbook_correct", False)),
        }
        return it["question"], it.get("id", f"q{int(idx):03d}"), gold

    def gate_idxs(self, run_dir, size, seed):
        """Frozen gate subsample (sampled once, persisted) -- identical
        persistence to Substrate.gate_idxs. Capped at the gate-pool size for the
        small MCQ set."""
        path = os.path.join(run_dir, "gate_idxs.json")
        if os.path.exists(path):
            idxs = json.load(open(path))
            print(f"[qa_substrate] gate: {len(idxs)} frozen idxs from {path}")
            return idxs
        size = min(int(size), len(self._gate_pool))
        idxs = sorted(random.Random(seed).sample(self._gate_pool, size))
        os.makedirs(run_dir, exist_ok=True)
        json.dump(idxs, open(path, "w"))
        print(f"[qa_substrate] gate: sampled {len(idxs)} idxs (seed {seed}) -> {path}")
        return idxs

    def rotating_gate_idxs(self, size, seed, epoch):
        """Resampled gate subsample, deterministic per (seed, epoch)."""
        size = min(int(size), len(self._gate_pool))
        rng = random.Random(seed * 100003 + epoch)
        return sorted(rng.sample(self._gate_pool, size))

    def make_evaluator(self):
        """Parity with Substrate.make_evaluator. MCQ scoring lives in McqReward
        (it needs the answerer + the SearchTarget), so there is no standalone
        evaluator object here."""
        return None

    @property
    def dataset_path(self):
        return self._path

    def eval_log_rows(self, recap_rows, step=None, candidate_sha=None):
        """Shape McqReward's per-question recap rows into flat, MLflow-table /
        Parquet-friendly columns (one row per question x candidate). Pure data
        reshaping -- no scoring."""
        out = []
        for r in recap_rows:
            row = {
                "step": step,
                "candidate_sha": candidate_sha,
                "q_id": r.get("q_id"),
                "question": r.get("question"),
                "chosen_idx": r.get("chosen_idx"),
                "gold_idx": r.get("gold_idx"),
                "openbook_correct": int(bool(r.get("openbook_correct"))),
                "closedbook_correct": int(bool(r.get("closedbook_correct"))),
                "retrieval_hit": int(bool(r.get("retrieval_hit"))),
                "error": r.get("error"),
                "context_preview": (r.get("context_preview") or "")[:300],
            }
            out.append(row)
        return out
