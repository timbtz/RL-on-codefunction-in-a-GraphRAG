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
from ..env.openai_client import BudgetExceeded, step_cost_delta
from ..env.sandbox import SandboxError
from ..reward.objectives import QUALITY_KEYS
from .gate import Gate
from .pool import CandidatePool
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
                 tracker, archive, budget=None):
        self._cfg = cfg
        self._graph = graph
        self._sandbox = sandbox
        self._reward = reward
        self._mutator = mutator
        self._edit_budget = edit_budget
        self._tracker = tracker
        self._archive = archive
        self._budget = budget  # OpenAIBudget (or None); per-step cost attribution
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
        # Phase B1/C2 attribution: split the failure into GENERATION vs RANKING
        # from recall@100 (gold reachable anywhere in the program's final dict)
        # vs recall@20. recall@100 high but recall@20 low => gold was generated
        # but mis-ranked (fix scoring/rerank); recall@100 low => gold never
        # generated (fix generation: reformulate/expand). empty_result flags a
        # dead traversal -- the program returned (almost) nothing for this query.
        r100 = r.get("recall@100", 0.0)
        r20 = r["metrics"].get("recall@20", 0.0)
        if r100 >= 0.999:
            attribution = "RANKING (gold reachable in top-100, fix scoring/rerank)"
        elif r100 > r20 + 1e-9:
            attribution = (f"MIXED (recall@100={r100:.2f} > recall@20={r20:.2f}: "
                           "some gold generated but unranked, some not generated)")
        else:
            attribution = ("GENERATION (gold not in top-100, fix generation: "
                           "reformulate / expand / broaden retrieval)")
        return {
            "query": r["query"],
            "bucket": bucket,
            "metrics": r["metrics"],
            "recall@100": r100,
            "attribution": attribution,
            "empty_result": len(retrieved) == 0,
            "missed_gold": [(i, texts.get(i, "")[:200]) for i in missed_ids],
            "top_wrong": [(i, s, texts.get(i, "")[:200]) for i, s in wrong],
            "gold_ranks": ranks[:5],
            "error": r.get("error"),
        }

    def _gate_value(self, mv):
        """Scalar that mirrors the gate's accept semantics -- used for the
        minibatch pre-screen and for choosing the pool's headline best."""
        if self._gate.mode == "strict":
            return mv.get(self._gate.metric)
        if self._gate.mode == "blend":
            return self._gate.composite(mv)
        return mv.primary  # dominance: fall back to recall@20 for a scalar view

    @staticmethod
    def _is_stop_progress(pool_on, admitted, accepted):
        """Phase 2: the stop counter resets on any ADMISSION when the pool drives
        the search (a new Pareto-frontier point OR a sole-best specialist), else on
        a headline gate accept. This is deliberately DECOUPLED from `frontier_grew`
        (which still drives architect escalation): run-7 step 27 was a real sole-best
        admission that nonetheless drove `stale` 6->7->8 and stopped the run at 29/40
        because the stop keyed off frontier growth. -> True iff this step made
        progress for the purpose of the stop condition."""
        return bool(admitted if pool_on else accepted)

    def _final_select(self, substrate, pool, best_prog, best, run_dir, get_fn):
        """Phase 1 -- final held-out bake-off. The rotating gate fights overfit but
        makes per-epoch scores incomparable, so the monotone incumbent is the "last
        belt-holder," not necessarily the best program. Re-score every surviving pool
        member on ONE fixed held-out set (the fenced meta-holdout the gate never
        touches) and export the argmax of the blend value -- OpenEvolve's
        get_best_program, but re-scored because our gate rotates.

        -> (export_prog, export_mv, scored), where scored is the full
        [(holdout_value, program, mv)] list (empty => caller keeps the incumbent).
        Also writes select_holdout.json next to best_search.py (no silent pick)."""
        cfg = self._cfg
        sel_idxs = list(getattr(substrate, "meta_holdout_idxs", []) or [])
        if not sel_idxs:                    # holdout disabled -> fixed gate fallback
            sel_idxs = substrate.gate_idxs(run_dir, cfg.gate_size, cfg.gate_seed)
        n = int(getattr(cfg, "select_holdout_n", 0) or 0)
        if n and n < len(sel_idxs):         # optional subsample to cap cost/latency
            sel_idxs = sorted(random.Random(cfg.gate_seed).sample(list(sel_idxs), n))
        # frontier-only keeps the bake-off bounded (typically <= cap=24); a specialist
        # that wins no holdout query cannot be the headline best anyway.
        cands = [m.program for m in pool.members] or [best_prog]
        ceiling = float(getattr(cfg, "openai_budget_usd", 0.0) or 0.0)
        print(f"[fast_loop] final bake-off: re-scoring {len(cands)} pool member(s) "
              f"on {len(sel_idxs)} fixed holdout queries (vs the rotating gate) ...")
        scored = []
        for p in cands:
            # Budget guard: a BudgetExceeded raised inside a program is swallowed by
            # the sandbox as a per-query crash (it never propagates out of score), so
            # we PROACTIVELY check the ceiling before each member and fall back to the
            # best scored so far. The except below is defensive in case it ever does.
            if (self._budget is not None and ceiling
                    and self._budget.spent_usd >= ceiling):
                print(f"[fast_loop] final bake-off: budget ${self._budget.spent_usd:.2f} "
                      f">= ${ceiling:.2f} ceiling after {len(scored)} member(s) -- "
                      f"stopping, falling back to best scored so far")
                break
            try:
                mv = self._reward.score(get_fn(p), sel_idxs, src=p.src,
                                        per_query_timeout_s=cfg.probe_timeout_s)
            except BudgetExceeded as e:
                print(f"[fast_loop] final bake-off: BudgetExceeded ({e}) -- "
                      f"falling back to best scored so far")
                break
            except Exception as e:          # a member that crashes the holdout is not best
                print(f"[fast_loop] final bake-off: member {p.sha[:8]} raised on "
                      f"holdout ({type(e).__name__}: {e}) -- excluded")
                continue
            val = self._gate_value(mv)
            print(f"[fast_loop]   {p.sha[:8]}  holdout {self._gate.mode}-value="
                  f"{val:.4f}  recall@20={mv.get('recall@20'):.4f}  "
                  f"cc={getattr(mv, 'code_complexity', 0.0):.0f}"
                  + ("  CRASHED" if getattr(mv, 'crashed', False) else ""))
            scored.append((val, p, mv))
        if not scored:
            print("[fast_loop] final bake-off: no member scored -- keeping the "
                  "rotating-gate incumbent as best")
            return best_prog, best, []
        # `scored` is ordered by holdout blend value (desc), tiebroken on the
        # simpler program by code_complexity (asc) -- this is the reported quality
        # ranking + the audit order. NOT llm_calls (the post-mortem found it inert
        # in run-7, flat ~3.1/query); code_complexity is a real, varying signal.
        # NOTE: code_complexity / rerank_items are top-level MetricVector
        # ATTRIBUTES -- mv.get() only reads the quality dict, so read via getattr.
        scored.sort(key=lambda t: (-t[0], getattr(t[2], "code_complexity", float("inf"))))
        quality_top = scored[0]
        export_prog, export_mv = self._cost_aware_pick(scored)
        self._write_select_audit(run_dir, sel_idxs, scored, export_prog, best_prog,
                                 quality_top[1])
        same = export_prog.sha == best_prog.sha
        cost_repick = export_prog.sha != quality_top[1].sha
        print(f"[fast_loop] final bake-off: export {export_prog.sha[:8]} "
              f"(holdout value={self._gate_value(export_mv):.4f}, "
              f"rerank_items={getattr(export_mv, 'rerank_items', 0.0):.1f}); "
              f"rotating-gate headline was {best_prog.sha[:8]} -- "
              + ("SAME program" if same else "DIFFERENT (the artifact was not the true best)")
              + (f"; cost-aware re-pick chose it over quality-top "
                 f"{quality_top[1].sha[:8]} within floor" if cost_repick else ""))
        return export_prog, export_mv, scored

    def _cost_aware_pick(self, scored):
        """0.6b / post-mortem #5 -- cost-aware EXPORT re-pick. `scored` is
        [(value, prog, mv)] sorted by holdout quality (desc). Among finalists
        within `select_cost_floor` of the top quality value, ship the CHEAPEST by
        the deterministic `rerank_items` cost meter, tiebreaking on code_complexity
        then quality. The accept gate stays PURE-QUALITY -- this only re-picks what
        SHIPS, so the (noise-free but still secondary) cost axis can never corrupt
        the search itself (the Option-2 decoupled structure the post-mortem ships).
        `select_cost_floor`=0 (default) => the band is the exact top value, so this
        degrades to the quality-argmax + code_complexity tiebreak (run-7 parity).
        -> (export_prog, export_mv)."""
        top = scored[0][0]
        floor = float(getattr(self._cfg, "select_cost_floor", 0.0) or 0.0)
        finalists = [t for t in scored if t[0] >= top - floor]
        pick = min(finalists, key=lambda t: (
            getattr(t[2], "rerank_items", 0.0),
            getattr(t[2], "code_complexity", float("inf")),
            -t[0]))
        return pick[1], pick[2]

    def _write_select_audit(self, run_dir, sel_idxs, scored, export_prog,
                            gate_headline_prog, quality_top_prog=None):
        """select_holdout.json: every surviving member's holdout blend value AND
        rerank_items cost meter (no silent pick) + which sha we exported vs the
        rotating-gate headline vs the pure-quality top (cost re-pick audit)."""
        members = [{
            "sha": p.sha,
            "holdout_value": round(val, 6),
            "recall@20": round(mv.get("recall@20"), 6),
            "hit@1": round(mv.get("hit@1"), 6),
            "mrr": round(mv.get("mrr"), 6),
            "rerank_items": round(getattr(mv, "rerank_items", 0.0), 3),
            "code_complexity": round(getattr(mv, "code_complexity", 0.0), 3),
            "crashed": bool(getattr(mv, "crashed", False)),
        } for val, p, mv in scored]
        audit = {
            "select_set_size": len(sel_idxs),
            "select_cost_floor": float(getattr(self._cfg, "select_cost_floor", 0.0) or 0.0),
            "exported_sha": export_prog.sha,
            "quality_top_sha": (quality_top_prog or export_prog).sha,
            "cost_repick": bool(quality_top_prog is not None
                                and export_prog.sha != quality_top_prog.sha),
            "gate_headline_sha": gate_headline_prog.sha,
            "export_changed": export_prog.sha != gate_headline_prog.sha,
            "members": members,
        }
        with open(os.path.join(run_dir, "select_holdout.json"), "w") as f:
            json.dump(audit, f, indent=1)

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

        # --- run-6 search controls -------------------------------------------
        pool_on = bool(getattr(cfg, "pool_enabled", False))
        pool_cap = int(getattr(cfg, "pool_cap", 24) or 24)
        mb_size = int(getattr(cfg, "minibatch_size", 0) or 0)
        meta_every = int(getattr(cfg, "meta_eval_every", 0) or 0)
        meta_idxs = list(getattr(substrate, "meta_holdout_idxs", []) or [])
        pool = CandidatePool(cap=pool_cap)

        # The headline incumbent (best_prog/best_fn/best) -- still a single
        # monotone gate-best for reporting/saving -- coexists with the pool, which
        # only drives parent selection and the frontier-stall stop (Phase A).
        best_prog = seed_program
        best_fn = self._sandbox.compile(best_prog.src)
        self._sandbox.probe(best_fn, probes, cfg.probe_timeout_s)
        fn_cache = {best_prog.sha: best_fn}

        def _get_fn(p):
            f = fn_cache.get(p.sha)
            if f is None:
                f = fn_cache[p.sha] = self._sandbox.compile(p.src)
            return f

        t0 = time.time()
        best = cache.get(_ckey(best_prog.sha))
        if best is None:
            best = self._reward.score(best_fn, gate_idxs, src=best_prog.src,
                                      per_query_timeout_s=cfg.probe_timeout_s)
            cache.put(_ckey(best_prog.sha), best)
        self._archive.add(best_prog.sha, best_prog.family, best)
        pool.consider(best_prog, best)
        print(f"[fast_loop] seed gate ({time.time()-t0:.0f}s): "
              f"{ {k: round(best.get(k), 4) for k in QUALITY_KEYS} }"
              + (f"  [pool ON cap={pool_cap}]" if pool_on else "  [single incumbent]"))
        self._tracker.log_vector("best_", best, step=-1)

        # Plateau handling: escalate to the architect tier and stop the campaign
        # after `stop_stale` STALE steps. "stale" now means steps adding no new
        # Pareto-front member (Phase A4 -- the direct fix for the run-5 step-18
        # stall); with the pool off it falls back to consecutive non-accepts.
        arch_plateau = int(getattr(cfg, "architect_plateau", 0) or 0)
        stop_stale = int(getattr(cfg, "stop_after_stale", 0) or 0)
        # `stale` (frontier-stall) drives ARCHITECT escalation; `stop_stale_ctr`
        # drives the STOP and resets on ANY admission, incl. sole-best specialists
        # (Phase 2 -- the two signals are no longer conflated).
        stale = 0
        stop_stale_ctr = 0

        # --- auditability baselines (Phase B cost split, Phase C lineage) ----
        # Snapshot the budget AFTER seed scoring + embedder warmup so warmup spend
        # lands in the baseline, not in step 0's delta (cost-boundary caveat).
        ceiling = float(getattr(cfg, "openai_budget_usd", 0.0) or 0.0)
        cost_prev = self._budget.snapshot() if self._budget is not None else None
        calls_prev = dict(self._mutator.call_counts)
        lineage_fh = open(os.path.join(run_dir, "lineage.jsonl"), "w")
        n_accepted, steps_run = 0, 0

        for step in range(steps):
            t_step = time.time()
            g_idxs, g_tag = _gate_for(step)
            if g_tag != gate_tag:
                # Gate rotated: re-score the headline incumbent on the new
                # subsample so the candidate-vs-incumbent comparison is on
                # identical queries (pool members keep their own-epoch scores).
                gate_idxs, gate_tag = g_idxs, g_tag
                rb = cache.get(_ckey(best_prog.sha))
                if rb is None:
                    rb = self._reward.score(best_fn, gate_idxs, src=best_prog.src,
                                            per_query_timeout_s=cfg.probe_timeout_s)
                    cache.put(_ckey(best_prog.sha), rb)
                best = rb
                print(f"[fast_loop] gate rotated -> {gate_tag}  "
                      f"incumbent recall@20={best.get('recall@20'):.4f} "
                      f"mrr={best.get('mrr'):.4f}")

            # Phase A3: pick the parent from the pool (sole-best-weighted) instead
            # of always descending the single incumbent.
            if pool_on and len(pool):
                parent_rng = random.Random(cfg.gate_seed * 911 + step)
                parent_prog = pool.select_parent(
                    parent_rng, discount=bool(getattr(cfg, "pool_discount", True))).program
            else:
                parent_prog = best_prog
            parent_sha = parent_prog.sha
            parent_fn = _get_fn(parent_prog)

            ridxs = random.Random(cfg.gate_seed + step).sample(
                substrate.train_idxs, cfg.rollout_batch)
            _, rows = self._reward.score(parent_fn, ridxs, src=parent_prog.src,
                                         return_rows=True,
                                         per_query_timeout_s=cfg.probe_timeout_s)
            fails, wins = self._reflect(rows, cfg.reflect_top)

            L_t = self._edit_budget.L_t(step)
            plateau = bool(arch_plateau and stale >= arch_plateau)
            t_llm = time.time()
            try:
                cand, transcript = self._mutator.propose(
                    parent_prog, fails, wins, buffer.recent(), L_t, plateau=plateau)
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
                            "minibatch_skipped": 0,
                            "llm_seconds": llm_seconds, "score_seconds": 0.0,
                            "edit_budget": L_t}
            reason, s = "", None
            admitted, frontier_grew = False, False
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
                    # Phase B2 + 0.5: cheap minibatch pre-screen -- only pay the
                    # full gate if the child clears the INCUMBENT `best` (not the
                    # sampled parent) within tolerance on a b-query subsample.
                    if mb_size and mb_size < len(gate_idxs) and not self._minibatch_ok(
                            cache, best_prog, best_fn, cand, cand_fn,
                            gate_idxs, gate_tag, mb_size, cfg):
                        step_metrics["minibatch_skipped"] = 1
                        reason = (f"failed minibatch pre-screen (b={mb_size}, did not "
                                  f"clear incumbent within eps)")
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
                        fn_cache[cand.sha] = cand_fn
                        frontier_grew, admitted = pool.consider(cand, s)
                        # headline gate (blend) -- updates the monotone reported best.
                        if self._gate.accept(s, best):
                            best_prog, best_fn, best = cand, cand_fn, s
                            cand.save(os.path.join(acc_dir, f"step_{step:03d}.py"))
                        cap = self._gate.max_complexity
                        if s.crashed:
                            reason = "crashed on gate"
                        elif cap and s.code_complexity > cap:
                            reason = (f"complexity cap exceeded "
                                      f"({s.code_complexity:.0f} > {cap:.0f}) -- simplify")
                        elif admitted:
                            reason = ("admitted to pool ("
                                      + ("new Pareto frontier" if frontier_grew
                                         else "sole-best on >=1 query") + ")")
                        else:
                            reason = f"not admitted (dominated, no sole-best query; gate {self._gate.mode})"

            # "accepted" = useful edit: pool admission when the pool drives the
            # search, else a headline gate improvement (run-5 parity).
            accepted = admitted if pool_on else (s is not None and best_prog.sha == cand.sha)
            progress = frontier_grew if pool_on else accepted
            step_metrics["accepted"] = int(bool(accepted))

            if progress:
                stale = 0
            else:
                stale += 1
            # Phase 2: the stop counter is decoupled from frontier growth -- it
            # resets on any admission (frontier OR sole-best), so a real specialist
            # admission no longer counts toward the stop.
            if self._is_stop_progress(pool_on, admitted, accepted):
                stop_stale_ctr = 0
            else:
                stop_stale_ctr += 1
            if not accepted:
                if s is not None:
                    delta = (f"recall@20 {s.get('recall@20') - best.get('recall@20'):+.4f}, "
                             f"mrr {s.get('mrr') - best.get('mrr'):+.4f}")
                else:
                    delta = "no gate score"
                gist = parent_prog.change_summary(cand) if cand else "(no candidate)"
                buffer.add(step, f"{gist} => {delta}; {reason}")
                buffer.save(buf_path)

            # per-step cost split (Phase B): OpenAI spend delta, accept-vs-reject,
            # plus per-step LLM call counts (the cumulative calls_* go at run end).
            if cost_prev is not None:
                cost_cur = self._budget.snapshot()
                step_metrics.update(step_cost_delta(
                    cost_prev, cost_cur, bool(accepted), ceiling))
                cost_prev = cost_cur
            calls_cur = dict(self._mutator.call_counts)
            for m in set(calls_cur) | set(calls_prev):
                d = calls_cur.get(m, 0) - calls_prev.get(m, 0)
                if d:
                    step_metrics[f"calls_{m}_step"] = d
            calls_prev = calls_cur
            if accepted:
                n_accepted += 1
            steps_run += 1

            # Phase B3: every `meta_every` accepts, score the headline best on the
            # fenced meta-holdout and log gate - meta (the overfit detector; it
            # does not gate). Skipped when the holdout is empty (size=0).
            meta_delta = None
            if meta_every and meta_idxs and accepted and (n_accepted % meta_every == 0):
                meta_mv = self._reward.score(best_fn, meta_idxs, src=best_prog.src,
                                             per_query_timeout_s=cfg.probe_timeout_s)
                meta_delta = self._gate_value(best) - self._gate_value(meta_mv)
                step_metrics.update({"meta_holdout_value": self._gate_value(meta_mv),
                                     "gate_minus_meta": meta_delta,
                                     "meta_recall_at_20": meta_mv.get("recall@20")})

            # consolidated lineage row (Phase C): one per step, accepted AND
            # rejected -- a rejected edit tells the proposer what not to retry.
            change_summary = (parent_prog.change_summary(cand) if cand
                              else "(no candidate)")
            tier = "architect" if plateau else "editor"
            lineage_fh.write(json.dumps({
                "step": step,
                "parent_sha": parent_sha,
                "child_sha": cand.sha if cand else None,
                "change_summary": change_summary,
                "metric_vector": s.as_flat() if s is not None else None,
                "accepted": bool(accepted),
                "admitted": bool(admitted),
                "frontier_grew": bool(frontier_grew),
                "pool_size": len(pool),
                "reason": reason or "accepted",
                "gate_minus_meta": meta_delta,
                "tokens_step": (step_metrics.get("tokens_accepted", 0)
                                + step_metrics.get("tokens_rejected", 0)),
                "edit_budget": L_t,
                "gate_tag": gate_tag,
            }) + "\n")
            lineage_fh.flush()

            # Phase G (gated): mirror that same row into an MLflow trace so the
            # decision is navigable in the Traces tab -- reflection in (inputs),
            # verdict + transcript out. No-op unless MLFLOW_TRACE_STEPS=1.
            verdict = "accept" if accepted else "reject"
            with self._tracker.step_span(f"step_{step:03d}_{verdict}", inputs={
                "parent_sha": parent_sha,
                "edit_budget": L_t,
                "tier": tier,
                "failures": [{"query": f["query"], "bucket": f["bucket"],
                              "attribution": f["attribution"]} for f in fails],
            }) as span:
                self._tracker.record_step(span, outputs={
                    "accepted": bool(accepted),
                    "reason": reason or "accepted",
                    "change_summary": change_summary,
                    "metric_vector": s.as_flat() if s is not None else None,
                    "transcript": transcript[:4000],
                }, attributes={
                    "step": step,
                    "child_sha": cand.sha if cand else None,
                    "admitted": bool(admitted),
                    "frontier_grew": bool(frontier_grew),
                    "gate_tag": gate_tag,
                    "llm_seconds": round(llm_seconds, 1),
                    "step_seconds": round(time.time() - t_step, 1),
                })

            self._tracker.log_vector("best_", best, step=step)
            self._tracker.log_metrics(step_metrics, step=step)
            status = "ACCEPT" if accepted else f"reject ({reason})"
            cand_r = f"{s.get('recall@20'):.4f}" if s else "-"
            print(f"[fast_loop] step {step:02d} {status}  cand recall@20={cand_r} "
                  f"best={best.get('recall@20'):.4f}  L={L_t} {tier} "
                  f"pool={len(pool)} stale={stale} stop_stale={stop_stale_ctr}  "
                  f"({time.time()-t_step:.0f}s)")
            if stop_stale and stop_stale_ctr >= stop_stale:
                kind = "no admission (frontier or sole-best)" if pool_on else "non-accepts"
                print(f"[fast_loop] STOPPING at step {step}: {stop_stale_ctr} consecutive "
                      f"stale steps ({kind}, >= stop_after_stale={stop_stale})", flush=True)
                self._tracker.log_metrics({"stopped_stale_at": step}, step=step)
                break

        lineage_fh.close()
        # run-end rollup inputs: accepted_total + usd_total feed $/accepted-edit.
        summary = {"accepted_total": n_accepted, "steps_run": steps_run,
                   "pool_size_final": len(pool)}
        if self._budget is not None:
            summary["usd_total"] = self._budget.spent_usd
        self._tracker.log_metrics(summary)

        cc = self._mutator.call_counts
        if cc:
            self._tracker.log_metrics({f"calls_{m}": n for m, n in cc.items()})
            print(f"[fast_loop] LLM calls by model: {dict(cc)}")
        # Phase 1: final held-out bake-off. Rotation makes per-epoch scores
        # incomparable, so the monotone incumbent ("last belt-holder") may not be the
        # true best. Re-score every surviving pool member on the fixed meta-holdout and
        # export the argmax; keep the rotating-gate incumbent as best_gate_headline.py
        # so we can measure how often the artifact != the true best.
        gate_headline_prog = best_prog
        sel_prog, sel_mv, scored = self._final_select(
            substrate, pool, best_prog, best, run_dir, _get_fn)
        if scored:
            gate_headline_prog.save(os.path.join(run_dir, "best_gate_headline.py"))
            best_prog, best = sel_prog, sel_mv
        best_prog.save(os.path.join(run_dir, "best_search.py"))
        seed_program.save(os.path.join(run_dir, "seed_used.py"))
        with open(os.path.join(run_dir, "seed_vs_best.diff"), "w") as f:
            f.write(seed_program.diff(best_prog, max_lines=10_000))
        print(f"[fast_loop] done. best: { {k: round(best.get(k), 4) for k in QUALITY_KEYS} }")
        return ArmResult(best_prog.family, best_prog, best, run_dir)

    def _minibatch_eps(self, mb_size):
        """Tolerance band for the pre-screen. `minibatch_eps` if set, else 1/b
        (~0.05 at b=20) -- one query's worth of the subsample, the smallest
        meaningful resolution on a b-query draw (post-mortem #4)."""
        return (float(getattr(self._cfg, "minibatch_eps", 0.0) or 0.0)
                or (1.0 / mb_size if mb_size else 0.0))

    def _minibatch_ok(self, cache, best_prog, best_fn, cand, cand_fn,
                      gate_idxs, gate_tag, mb_size, cfg):
        """Phase 0.5 / post-mortem #4: cheap b-query pre-screen before paying the
        full gate. Two fixes over the run-7 screen, which false-killed 63% of
        candidates (19/30):
          1. Reference the monotone INCUMBENT `best` -- the same bar the full gate
             uses -- NOT a randomly-sampled pool parent (a specialist parent is a
             stricter, stochastic bar than the gate ever applies).
          2. Promote on `value(child) >= value(best) - eps`, NOT strict `>`: a
             20-query subsample has SE ~ 0.11, so strict-greater false-kills exact
             ties and primary-axis-positive children that the deciding gate would
             have accepted.
        Leakage-free: mb_idxs is a subset of the gate set, and a pass only sends
        MORE candidates to the strict full gate (which stays authoritative).
        Incumbent + child minibatch scores are cached per (sha, epoch).
        -> True if the child should proceed to the full gate."""
        # deterministic per-epoch subsample (no hash() -- that is process-salted)
        tag_seed = cfg.gate_seed * 13 + sum(ord(c) for c in gate_tag)
        mb_idxs = sorted(random.Random(tag_seed).sample(list(gate_idxs), mb_size))
        bk = f"mb:{best_prog.sha}@{gate_tag}"
        b_mb = cache.get(bk)
        if b_mb is None:
            b_mb = self._reward.score(best_fn, mb_idxs, src=best_prog.src,
                                      per_query_timeout_s=cfg.probe_timeout_s)
            cache.put(bk, b_mb)
        c_mb = self._reward.score(cand_fn, mb_idxs, src=cand.src,
                                  per_query_timeout_s=cfg.probe_timeout_s)
        cache.put(f"mb:{cand.sha}@{gate_tag}", c_mb)
        return self._gate_value(c_mb) >= self._gate_value(b_mb) - self._minibatch_eps(mb_size)
