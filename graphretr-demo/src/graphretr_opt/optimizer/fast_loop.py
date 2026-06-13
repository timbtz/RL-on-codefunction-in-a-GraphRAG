"""FastLoop -- descends ONE strategy basin. This is the SkillOpt step sequence
and the whole of the Stage-1 demo: rollout -> reflect -> mutate -> gate ->
accept/reject, with a fixed-subsample gate, an sha-keyed score cache, and a
rejected buffer feeding the next prompt.

It never touches the test split. All cross-strategy machinery (SlowLoop,
scheduler) is out of scope here -- FastLoop takes a seed and a budget and
returns an ArmResult.
"""
import json
import os
import random
import time
from dataclasses import dataclass

from ..agents.single import AgentUnavailable
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
                 tracker, archive):
        self._cfg = cfg
        self._graph = graph
        self._sandbox = sandbox
        self._reward = reward
        self._mutator = mutator
        self._edit_budget = edit_budget
        self._tracker = tracker
        self._archive = archive
        blend = {}
        for part in str(getattr(cfg, "gate_blend", "") or "").split(","):
            if ":" in part:
                k, w = part.split(":", 1)
                blend[k.strip()] = float(w)
        self._gate = Gate(mode=getattr(cfg, "gate_mode", "strict"),
                          metric=cfg.gate_metric, blend=blend or None,
                          max_complexity=getattr(cfg, "gate_max_complexity", 0.0))

    def _probe_queries(self, substrate, n=3):
        idxs = random.Random(self._cfg.gate_seed).sample(substrate.train_idxs, n)
        return [substrate.example(i)[0] for i in idxs]

    def _reflect(self, rows, n):
        """Harvest two failure buckets + a few protected successes.

        - 'missed': gold absent from top-20 (a RETRIEVAL failure) -- sorted by
          worst recall.
        - 'misranked': gold IS in top-20 but not at rank 0 (a RANKING failure,
          the Hit@1 weakness) -- sorted by worst mrr.
        Splitting reflect_top ~50/50 teaches the optimizer both. Winners are
        currently-passing queries shown as 'do not regress' (success-path
        regularizer; cheap insurance now that the complexity cap is off).
        -> (failures, wins)."""
        def m(r):
            return r["metrics"]
        missed = [r for r in rows
                  if r.get("error") or m(r)["recall@20"] < 1.0]
        misranked = [r for r in rows if not r.get("error")
                     and m(r)["recall@20"] >= 1.0 and m(r)["hit@1"] < 1.0]
        missed.sort(key=lambda r: (m(r)["recall@20"], m(r)["mrr"]))
        misranked.sort(key=lambda r: m(r)["mrr"])
        half = max(1, n // 2)
        chosen = ([(r, "missed") for r in missed[:n - half]]
                  + [(r, "misranked") for r in misranked[:half]])
        failures = [self._failure_record(r, bucket) for r, bucket in chosen]
        wins = [{"query": r["query"], "metrics": m(r)} for r in rows
                if not r.get("error") and m(r)["hit@1"] >= 1.0][:3]
        return failures, wins

    def _failure_record(self, r, bucket):
        """One reflection entry: missed-gold texts, the top non-gold nodes that
        out-ranked the gold (with the program's scores), and where retrieved gold
        actually landed -- the signals run 3's prompt lacked (bare ids only)."""
        retrieved = r["retrieved"]                       # [(id, score)] top-20
        gold = set(r["answer_ids"])
        top_ids = {i for i, _ in retrieved}
        missed_ids = [a for a in r["answer_ids"] if a not in top_ids][:5]
        wrong = [(i, s) for i, s in retrieved if i not in gold][:5]
        need = missed_ids + [i for i, _ in wrong]
        texts = self._graph.get_text(need) if need else {}
        ranks = sorted((rank, sc, i) for i, (rank, sc) in r.get("gold_ranks", {}).items())
        return {
            "query": r["query"],
            "bucket": bucket,
            "metrics": r["metrics"],
            "missed_gold": [(i, texts.get(i, "")[:200]) for i in missed_ids],
            "top_wrong": [(i, s, texts.get(i, "")[:200]) for i, s in wrong],
            "gold_ranks": ranks[:5],
            "error": r.get("error"),
        }

    def run(self, substrate, seed_program, steps, campaign) -> ArmResult:
        cfg = self._cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        acc_dir = os.path.join(run_dir, "accepted")
        os.makedirs(acc_dir, exist_ok=True)

        rotate = int(getattr(cfg, "gate_rotate_every", 0) or 0)

        def _gate_for(step):
            if not rotate:
                return substrate.gate_idxs(run_dir, cfg.gate_size, cfg.gate_seed), "fix"
            epoch = step // rotate
            return substrate.rotating_gate_idxs(cfg.gate_size, cfg.gate_seed, epoch), f"e{epoch}"

        probes = self._probe_queries(substrate)
        cache = _ScoreCache(run_dir)
        buffer = RejectedBuffer(cfg.buffer_last).load(os.path.join(run_dir, "buffer.json"))
        buf_path = os.path.join(run_dir, "buffer.json")

        gate_idxs, gate_tag = _gate_for(0)

        # A candidate's gate score depends on WHICH subsample it was scored on, so
        # the score cache is keyed by (sha, gate epoch) when the gate rotates.
        def _ckey(sha):
            return sha if not rotate else f"{sha}@{gate_tag}"

        prog = seed_program
        fn = self._sandbox.compile(prog.src)
        self._sandbox.probe(fn, probes, cfg.probe_timeout_s)

        t0 = time.time()
        best = cache.get(_ckey(prog.sha))
        if best is None:
            best = self._reward.score(fn, gate_idxs, src=prog.src,
                                      per_query_timeout_s=cfg.probe_timeout_s)
            cache.put(_ckey(prog.sha), best)
        self._archive.add(prog.sha, prog.family, best)
        print(f"[fast_loop] seed gate ({time.time()-t0:.0f}s): "
              f"{ {k: round(best.get(k), 4) for k in QUALITY_KEYS} }")
        self._tracker.log_vector("best_", best, step=-1)

        # Plateau handling: escalate to the architect tier after `arch_plateau`
        # consecutive non-accepts, and stop the campaign after `stop_stale` of
        # them (0 = run all `steps`). run 3 was still improving when a fixed step
        # count cut it off -- stalling, not a counter, is the right stop signal.
        arch_plateau = int(getattr(cfg, "architect_plateau", 0) or 0)
        stop_stale = int(getattr(cfg, "stop_after_stale", 0) or 0)
        stale = 0

        for step in range(steps):
            t_step = time.time()
            g_idxs, g_tag = _gate_for(step)
            if g_tag != gate_tag:
                # Gate rotated: re-score the incumbent on the new subsample so the
                # candidate-vs-incumbent comparison below is on identical queries.
                gate_idxs, gate_tag = g_idxs, g_tag
                rb = cache.get(_ckey(prog.sha))
                if rb is None:
                    rb = self._reward.score(fn, gate_idxs, src=prog.src,
                                            per_query_timeout_s=cfg.probe_timeout_s)
                    cache.put(_ckey(prog.sha), rb)
                best = rb
                print(f"[fast_loop] gate rotated -> {gate_tag}  "
                      f"incumbent recall@20={best.get('recall@20'):.4f} "
                      f"mrr={best.get('mrr'):.4f}")
            ridxs = random.Random(cfg.gate_seed + step).sample(
                substrate.train_idxs, cfg.rollout_batch)
            _, rows = self._reward.score(fn, ridxs, src=prog.src, return_rows=True,
                                         per_query_timeout_s=cfg.probe_timeout_s)
            fails, wins = self._reflect(rows, cfg.reflect_top)

            L_t = self._edit_budget.L_t(step)
            plateau = bool(arch_plateau and stale >= arch_plateau)
            t_llm = time.time()
            try:
                cand, transcript = self._mutator.propose(
                    prog, fails, wins, buffer.recent(), L_t, plateau=plateau)
            except AgentUnavailable as e:
                # CLI limits reached: stop the campaign here, keep the incumbent
                # as best, and fall through to the normal save-and-return path so
                # `cli final` can run on what we have.
                print(f"[fast_loop] STOPPING at step {step}: agent unavailable "
                      f"({e}) -- saving incumbent as best", flush=True)
                self._tracker.log_metrics({"stopped_early_at": step}, step=step)
                break
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
                    s = cache.get(_ckey(cand.sha))
                    if s is not None:
                        step_metrics["gate_cache_hit"] = 1
                    else:
                        s = self._reward.score(cand_fn, gate_idxs, src=cand.src,
                                               per_query_timeout_s=cfg.probe_timeout_s)
                        cache.put(_ckey(cand.sha), s)
                    step_metrics["score_seconds"] = time.time() - t_sc
                    self._tracker.log_vector("val_", s, step=step)
                    self._archive.add(cand.sha, cand.family, s)
                    if self._gate.accept(s, best):
                        step_metrics["accepted"] = 1
                        prog, fn, best = cand, cand_fn, s
                        cand.save(os.path.join(acc_dir, f"step_{step:03d}.py"))
                    else:
                        cap = self._gate.max_complexity
                        if s.crashed:
                            reason = "crashed on gate"
                        elif cap and s.code_complexity > cap:
                            reason = (f"complexity cap exceeded "
                                      f"({s.code_complexity:.0f} > {cap:.0f}) -- simplify")
                        else:
                            reason = f"gate not improved ({self._gate.mode})"

            if step_metrics["accepted"]:
                stale = 0
            else:
                stale += 1
                if s is not None:
                    delta = (f"recall@20 {s.get('recall@20') - best.get('recall@20'):+.4f}, "
                             f"mrr {s.get('mrr') - best.get('mrr'):+.4f}")
                else:
                    delta = "no gate score"
                gist = prog.change_summary(cand) if cand else "(no candidate)"
                buffer.add(step, f"{gist} => {delta}; {reason}")
                buffer.save(buf_path)

            self._tracker.log_vector("best_", best, step=step)
            self._tracker.log_metrics(step_metrics, step=step)
            status = "ACCEPT" if step_metrics["accepted"] else f"reject ({reason})"
            cand_r = f"{s.get('recall@20'):.4f}" if s else "-"
            tier = "architect" if plateau else "editor"
            print(f"[fast_loop] step {step:02d} {status}  cand recall@20={cand_r} "
                  f"best={best.get('recall@20'):.4f}  L={L_t} {tier} stale={stale}  "
                  f"({time.time()-t_step:.0f}s)")
            if stop_stale and stale >= stop_stale:
                print(f"[fast_loop] STOPPING at step {step}: {stale} consecutive "
                      f"non-accepts (>= stop_after_stale={stop_stale})", flush=True)
                self._tracker.log_metrics({"stopped_stale_at": step}, step=step)
                break

        cc = self._mutator.call_counts
        if cc:
            self._tracker.log_metrics({f"calls_{m}": n for m, n in cc.items()})
            print(f"[fast_loop] LLM calls by model: {dict(cc)}")
        prog.save(os.path.join(run_dir, "best_search.py"))
        seed_program.save(os.path.join(run_dir, "seed_used.py"))
        with open(os.path.join(run_dir, "seed_vs_best.diff"), "w") as f:
            f.write(seed_program.diff(prog, max_lines=10_000))
        print(f"[fast_loop] done. best: { {k: round(best.get(k), 4) for k in QUALITY_KEYS} }")
        return ArmResult(prog.family, prog, best, run_dir)
