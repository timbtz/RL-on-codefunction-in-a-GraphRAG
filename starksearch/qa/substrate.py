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
    def __init__(self, meta_holdout_size=0, meta_seed=1234,
                 promote_size=0, promote_seed=5678):
        print("[substrate] loading STaRK prime SKB + QA ...")
        self.skb = load_skb("prime", download_processed=True)
        self.qa = load_qa("prime")
        split = self.qa.get_idx_split()  # positional indices, NOT q_ids
        self._train = [int(i) for i in split["train"]]
        self._val = [int(i) for i in split["val"]]
        self._test = [int(i) for i in split["test"]]
        # Three disjoint slices over val (run10c cascaded eval):
        #   meta_holdout -- final export arbiter; the gate NEVER samples it. It
        #                   also detects overfit (the `gate - meta` gap).
        #   promote      -- mid-run promotion-confirm set (C); also fenced from the
        #                   gate AND disjoint from meta_holdout, so the repeated
        #                   promotion checks don't leak into the one-time export pick.
        #   gate_pool    -- everything else; the accept gate samples here.
        # size=0 on both leaves the whole val split as the gate pool (run-5
        # behaviour, so prior runs reproduce exactly).
        self._meta_holdout, self._promote, self._gate_pool = self._partition(
            meta_holdout_size, meta_seed, promote_size, promote_seed)
        print(f"[substrate] splits: train {len(self._train)} / val {len(self._val)} "
              f"(gate-pool {len(self._gate_pool)} / promote {len(self._promote)} / "
              f"meta-holdout {len(self._meta_holdout)}) / test {len(self._test)}")

    def _partition(self, mh_size, mh_seed, pr_size, pr_seed):
        """Disjoint (meta_holdout, promote, gate_pool) over val. The meta-holdout
        is carved first with the SAME seed expression as before, so its idxs are
        byte-identical when promote is off (prior runs reproduce exactly). The
        promote slice is then carved from the remainder, and gate_pool is the rest.
        size<=0 on a slice => that slice is empty."""
        holdout = set()
        if mh_size > 0:
            size = min(int(mh_size), len(self._val))
            holdout = set(random.Random(mh_seed * 7919 + 1).sample(self._val, size))
        rest = [i for i in self._val if i not in holdout]
        promote = set()
        if pr_size > 0:
            size = min(int(pr_size), len(rest))
            promote = set(random.Random(pr_seed * 7919 + 1).sample(rest, size))
        gate_pool = [i for i in rest if i not in promote]
        return sorted(holdout), sorted(promote), gate_pool

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

    @property
    def promote_idxs(self):
        """C: the dedicated promotion-confirm slice -- disjoint from the gate pool
        AND the meta-holdout arbiter. A gate-passing candidate is re-scored here
        before the exported headline best moves (n=100 noise guard)."""
        return list(self._promote)

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
