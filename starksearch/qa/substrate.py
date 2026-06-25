"""Substrate -- the QA/KB data layer: load_qa/load_skb, splits as python ints,
gold node-ID sets, the frozen val-gate subsample, and the STaRK Evaluator
factory.

The gate subsample is written to <run_dir>/gate_idxs.json on first use and
read back (never resampled) thereafter, so every step of a campaign -- and
every rerun -- scores the exact same val queries.

The test split lives behind get_test_idxs_I_KNOW_THIS_IS_FINAL(); only the
final-test entrypoint may call it.
"""
import json
import os
import random

from stark_qa import load_qa, load_skb
from stark_qa.evaluator import Evaluator


class Substrate:
    def __init__(self, meta_holdout_size=0, meta_seed=1234):
        print("[substrate] loading STaRK prime SKB + QA ...")
        self.skb = load_skb("prime", download_processed=True)
        self.qa = load_qa("prime")
        split = self.qa.get_idx_split()  # positional indices, NOT q_ids
        self._train = [int(i) for i in split["train"]]
        self._val = [int(i) for i in split["val"]]
        self._test = [int(i) for i in split["test"]]
        # Phase B3: fence off a fixed meta-validation slice of val that the
        # rotating gate NEVER samples. It does not gate -- it DETECTS overfit
        # (the `gate - meta` gap) -- and is the prerequisite guard for run-7's
        # editable prompts. size=0 leaves the whole val split as the gate pool
        # (run-5 behaviour, so prior runs reproduce exactly).
        self._meta_holdout, self._gate_pool = self._partition(meta_holdout_size, meta_seed)
        print(f"[substrate] splits: train {len(self._train)} / val {len(self._val)} "
              f"(gate-pool {len(self._gate_pool)} / meta-holdout {len(self._meta_holdout)}) "
              f"/ test {len(self._test)}")

    def _partition(self, size, seed):
        """Disjoint (meta_holdout, gate_pool) over val. size<=0 => empty holdout."""
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
        """B3: the fixed val slice the gate never touches (overfit detector)."""
        return list(self._meta_holdout)

    def example(self, idx):
        """Positional idx -> (query, q_id, answer_ids:list[int])."""
        query, q_id, answer_ids, _ = self.qa[int(idx)]
        return query, int(q_id), [int(a) for a in answer_ids]

    def gate_idxs(self, run_dir, size, seed):
        """Frozen val subsample for the accept gate (sampled once, persisted)."""
        path = os.path.join(run_dir, "gate_idxs.json")
        if os.path.exists(path):
            idxs = json.load(open(path))
            print(f"[substrate] gate: {len(idxs)} frozen idxs from {path}")
            return idxs
        idxs = sorted(random.Random(seed).sample(self._gate_pool, size))
        os.makedirs(run_dir, exist_ok=True)
        json.dump(idxs, open(path, "w"))
        print(f"[substrate] gate: sampled {len(idxs)} val idxs (seed {seed}) -> {path}")
        return idxs

    def rotating_gate_idxs(self, size, seed, epoch):
        """A resampled val subsample for a ROTATING accept gate -- deterministic
        per (seed, epoch) but different each epoch, so the optimizer cannot
        memorise one frozen 200-query subset. Not persisted; reproducible."""
        rng = random.Random(seed * 100003 + epoch)
        return sorted(rng.sample(self._gate_pool, size))

    def get_test_idxs_I_KNOW_THIS_IS_FINAL(self):
        """The locked test split. Called ONLY by the final-test entrypoint."""
        return list(self._test)

    def make_evaluator(self):
        return Evaluator([int(c) for c in self.skb.candidate_ids])
