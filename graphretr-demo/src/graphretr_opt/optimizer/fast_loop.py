"""FastLoop -- descends ONE strategy basin. This is the SkillOpt step sequence
and the whole of the Stage-1 demo: rollout -> reflect -> mutate -> gate ->
accept/reject, with a fixed-subsample gate, an sha-keyed score cache, and a
rejected buffer feeding the next prompt.

It never touches the test split. All cross-strategy machinery (SlowLoop,
scheduler, momentum updates) is out of scope here -- FastLoop takes a seed and
a budget and returns an ArmResult.
"""
import json
import os
import random
import time
from dataclasses import dataclass

from ..artifact.program import SearchProgram
from ..env.sandbox import SandboxError
from ..reward.objectives import QUALITY_KEYS
from .gate import Gate
from .rejected_buffer import RejectedBuffer


@dataclass
class ArmResult:
    family: str
    best_program: SearchProgram
    best_metrics: object
    run_dir: str


def _qdict(mv):
    return {k: mv.get(k) for k in QUALITY_KEYS}


class _ScoreCache:
    """sha256(source) -> gate MetricVector, persisted per run dir."""

    def __init__(self, run_dir):
        self._path = os.path.join(run_dir, "score_cache.json")
        self._raw = json.load(open(self._path)) if os.path.exists(self._path) else {}
        self._mem = {}

    def get(self, sha):
        return self._mem.get(sha)

    def put(self, sha, mv):
        self._mem[sha] = mv
        self._raw[sha] = mv.as_flat()
        json.dump(self._raw, open(self._path, "w"), indent=1)


class FastLoop:
    def __init__(self, cfg, graph, sandbox, reward, mutator, edit_budget,
                 tracker, archive, momentum=None):
        self._cfg = cfg
        self._graph = graph
        self._sandbox = sandbox
        self._reward = reward
        self._mutator = mutator
        self._edit_budget = edit_budget
        self._tracker = tracker
        self._archive = archive
        self._momentum = momentum
        self._gate = Gate(mode="strict", metric=cfg.gate_metric)

    def _probe_queries(self, substrate, n=3):
        idxs = random.Random(self._cfg.gate_seed).sample(substrate.train_idxs, n)
        return [substrate.example(i)[0] for i in idxs]

    def _worst_failures(self, rows, n):
        misses = [r for r in rows if r.get("error") or r["metrics"]["recall@20"] < 1.0]
        misses.sort(key=lambda r: (r["metrics"]["recall@20"], r["metrics"]["mrr"]))
        out = []
        for r in misses[:n]:
            retrieved = set(r["retrieved"])
            missed_ids = [a for a in r["answer_ids"] if a not in retrieved][:5]
            texts = self._graph.get_text(missed_ids) if missed_ids else {}
            out.append({
                "query": r["query"],
                "recall@20": r["metrics"]["recall@20"],
                "hit@1": r["metrics"]["hit@1"],
                "mrr": r["metrics"]["mrr"],
                "gold_ids": r["answer_ids"][:20],
                "missed": [(i, texts.get(i, "")[:300]) for i in missed_ids],
                "retrieved": r["retrieved"],
                "error": r.get("error"),
            })
        return out

    def run(self, substrate, seed_program, steps, campaign) -> ArmResult:
        cfg = self._cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        acc_dir = os.path.join(run_dir, "accepted")
        os.makedirs(acc_dir, exist_ok=True)

        gate_idxs = substrate.gate_idxs(run_dir, cfg.gate_size, cfg.gate_seed)
        probes = self._probe_queries(substrate)
        cache = _ScoreCache(run_dir)
        buffer = RejectedBuffer(cfg.buffer_last).load(os.path.join(run_dir, "buffer.json"))
        buf_path = os.path.join(run_dir, "buffer.json")

        prog = seed_program
        fn = self._sandbox.compile(prog.src)
        self._sandbox.probe(fn, probes, cfg.probe_timeout_s)

        t0 = time.time()
        best = cache.get(prog.sha)
        if best is None:
            best = self._reward.score(fn, gate_idxs, src=prog.src,
                                      per_query_timeout_s=cfg.probe_timeout_s)
            cache.put(prog.sha, best)
        self._archive.add(prog.sha, prog.family, best)
        print(f"[fast_loop] seed gate ({time.time()-t0:.0f}s): "
              f"{ {k: round(best.get(k), 4) for k in QUALITY_KEYS} }")
        self._tracker.log_vector("best_", best, step=-1)

        for step in range(steps):
            t_step = time.time()
            ridxs = random.Random(cfg.gate_seed + step).sample(
                substrate.train_idxs, cfg.rollout_batch)
            _, rows = self._reward.score(fn, ridxs, src=prog.src, return_rows=True,
                                         per_query_timeout_s=cfg.probe_timeout_s)
            fails = self._worst_failures(rows, cfg.reflect_top)

            L_t = self._edit_budget.L_t(step)
            momentum_ctx = self._momentum.context() if self._momentum else ""
            t_llm = time.time()
            cand, transcript = self._mutator.propose(
                prog, fails, buffer.recent(), L_t, momentum_ctx)
            llm_seconds = time.time() - t_llm

            with open(os.path.join(run_dir, f"reflection_{step:03d}.md"), "w") as f:
                f.write(transcript)
            if cand is not None:
                cand.save(os.path.join(run_dir, f"step_{step:03d}.py"))

            step_metrics = {"accepted": 0, "probe_failed": 0, "gate_cache_hit": 0,
                            "llm_seconds": llm_seconds, "score_seconds": 0.0,
                            "edit_budget": L_t}
            reason, s = "", None
            if cand is None:
                reason = "mutator returned no usable candidate (budget/parse/agent failure)"
            else:
                try:
                    cand_fn = self._sandbox.compile(cand.src)
                    self._sandbox.probe(cand_fn, probes, cfg.probe_timeout_s)
                except SandboxError as e:
                    step_metrics["probe_failed"] = 1
                    reason = f"sandbox/probe rejected: {e}"
                else:
                    t_sc = time.time()
                    s = cache.get(cand.sha)
                    if s is not None:
                        step_metrics["gate_cache_hit"] = 1
                    else:
                        s = self._reward.score(cand_fn, gate_idxs, src=cand.src,
                                               per_query_timeout_s=cfg.probe_timeout_s)
                        cache.put(cand.sha, s)
                    step_metrics["score_seconds"] = time.time() - t_sc
                    self._tracker.log_vector("val_", s, step=step)
                    self._archive.add(cand.sha, cand.family, s)
                    if self._gate.accept(s, best):
                        step_metrics["accepted"] = 1
                        prog, fn, best = cand, cand_fn, s
                        cand.save(os.path.join(acc_dir, f"step_{step:03d}.py"))
                    else:
                        reason = ("crashed on gate" if s.crashed
                                  else "gate not improved (strict > on recall@20)")

            if step_metrics["accepted"] == 0:
                buffer.add(step, prog.diff(cand) if cand else "(no candidate)",
                           _qdict(best), (_qdict(s) if s else None), reason)
                buffer.save(buf_path)

            self._tracker.log_vector("best_", best, step=step)
            self._tracker.log_metrics(step_metrics, step=step)
            status = "ACCEPT" if step_metrics["accepted"] else f"reject ({reason})"
            cand_r = f"{s.get('recall@20'):.4f}" if s else "-"
            print(f"[fast_loop] step {step:02d} {status}  cand recall@20={cand_r} "
                  f"best={best.get('recall@20'):.4f}  L={L_t}  ({time.time()-t_step:.0f}s)")

        prog.save(os.path.join(run_dir, "best_search.py"))
        seed_program.save(os.path.join(run_dir, "seed_used.py"))
        with open(os.path.join(run_dir, "seed_vs_best.diff"), "w") as f:
            f.write(seed_program.diff(prog, max_lines=10_000))
        print(f"[fast_loop] done. best: { {k: round(best.get(k), 4) for k in QUALITY_KEYS} }")
        return ArmResult(prog.family, prog, best, run_dir)
