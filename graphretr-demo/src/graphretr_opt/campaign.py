"""Campaign -- the orchestrator. Owns all state and wires the two halves
(env + optimizer) for the three entrypoints: stage0-probe, optimize, final-test.

boot() assembles the immutable env (backend -> RetrievalGraph -> Sandbox) and
the data/reward stack once; each entrypoint composes the optimizer pieces it
needs on top. Stage-1 drives FastLoop directly and ignores SlowLoop/scheduler/
gepa_adapter (all present as seams).
"""
import os
import random
import time
from dataclasses import replace

from .config import load_config, load_strategy
from .data.substrate import Substrate
from .env.backends.falkordb import FalkorDBBackend
from .env.cache import PrimitiveCache
from .env.embedder import make_embedder
from .env.openai_client import OpenAIBudget
from .env.retrieval_graph import RetrievalGraph
from .env.sandbox import Sandbox, SandboxError, SAFE_BUILTINS
from .reward.evaluator import RewardModel
from .reward.objectives import QUALITY_KEYS
from .reward.pareto import ParetoArchive
from .artifact.program import SearchProgram
from .agents.single import SingleCoder
from .agents.team import TieredCoder
from .optimizer.edit_budget import EditBudget
from .optimizer.mutator import Mutator
from .optimizer.fast_loop import FastLoop
from .tracking.mlflow_tracker import MlflowTracker


class Campaign:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()

    # --------------------------------------------------------------- bring-up

    def boot(self):
        cfg = self.cfg
        backend = FalkorDBBackend(cfg.falkor_host, cfg.falkor_port, cfg.graph_name)
        # One shared metered OpenAI gateway for the embedder AND G.extract;
        # without a key both stay disabled and the minilm path works as before.
        budget = None
        if os.environ.get("OPENAI_API_KEY"):
            budget = OpenAIBudget(os.path.join(cfg.runs_dir, "openai_usage.json"),
                                  ceiling_usd=cfg.openai_budget_usd)
            print(f"[campaign] openai budget: ${budget.spent_usd:.2f} spent of "
                  f"${cfg.openai_budget_usd:.2f} ceiling (embedder={cfg.embedder})")
        self.graph = RetrievalGraph(cfg, backend, PrimitiveCache(),
                                    make_embedder(cfg, budget), llm_budget=budget)
        self.sandbox = Sandbox(self.graph, default_timeout_s=cfg.probe_timeout_s)
        self.substrate = Substrate()
        self.reward = RewardModel(self.substrate, self.sandbox, cfg.crash_frac_limit)
        self.archive = ParetoArchive()
        return self

    def _make_mutator(self):
        cfg = self.cfg
        if cfg.mutator_agent == "tiered":
            agent = TieredCoder(cfg.mutator_backend, cfg.analyst_model,
                                cfg.editor_model, cfg.architect_model,
                                cfg.llm_timeout_s)
        else:
            agent = SingleCoder(cfg.mutator_backend, cfg.mutator_model,
                                cfg.llm_timeout_s)
        return Mutator(agent, self.graph.describe(), SAFE_BUILTINS)

    def _seed_program(self):
        strat = load_strategy(self.cfg)
        return SearchProgram.from_file(strat["seed"], family=strat["family"])

    def _probe_queries(self, n=3):
        idxs = random.Random(self.cfg.gate_seed).sample(self.substrate.train_idxs, n)
        return [self.substrate.example(i)[0] for i in idxs]

    # ----------------------------------------------------------- entrypoints

    def optimize(self, steps=None, campaign="campaign"):
        cfg = self.cfg
        steps = steps or cfg.steps
        edit_budget = EditBudget(cfg.edit_schedule, cfg.max_edits, cfg.min_edits, steps)
        mutator = self._make_mutator()
        seed = self._seed_program()
        run_dir = os.path.join(cfg.runs_dir, campaign)
        os.makedirs(run_dir, exist_ok=True)

        with MlflowTracker(cfg).start(campaign, params={
            **{k: v for k, v in vars(cfg).items() if k != "root"},
            "campaign": campaign, "steps": steps, "git_sha": _git_sha(),
        }) as tracker:
            loop = FastLoop(cfg, self.graph, self.sandbox, self.reward, mutator,
                            edit_budget, tracker, self.archive)
            result = loop.run(self.substrate, seed, steps, campaign)
            tracker.log_artifacts(run_dir)
        print(f"[campaign] best program -> {os.path.join(run_dir, 'best_search.py')}")
        return result

    def stage0(self, campaign="stage0"):
        """Headroom check: score the seed, then ONE one-shot rewrite (no loop)."""
        cfg = self.cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        os.makedirs(run_dir, exist_ok=True)
        gate = self.substrate.gate_idxs(run_dir, cfg.gate_size, cfg.gate_seed)
        probes = self._probe_queries()
        seed = self._seed_program()
        seed_fn = self.sandbox.compile(seed.src)
        self.sandbox.probe(seed_fn, probes, cfg.probe_timeout_s)
        mutator = self._make_mutator()

        with MlflowTracker(cfg).start(campaign, params={
            k: v for k, v in vars(cfg).items() if k != "root"}) as tracker:
            t0 = time.time()
            seed_mv = self.reward.score(seed_fn, gate, src=seed.src,
                                        per_query_timeout_s=cfg.probe_timeout_s)
            print(f"[stage0] seed ({time.time()-t0:.0f}s): "
                  f"{ {k: round(seed_mv.get(k), 4) for k in QUALITY_KEYS} }")
            tracker.log_vector("seed_", seed_mv)

            ridxs = random.Random(cfg.gate_seed).sample(
                self.substrate.train_idxs, cfg.rollout_batch)
            _, rows = self.reward.score(seed_fn, ridxs, src=seed.src, return_rows=True,
                                        per_query_timeout_s=cfg.probe_timeout_s)
            loop = FastLoop(cfg, self.graph, self.sandbox, self.reward, mutator,
                            EditBudget("const", cfg.max_edits), tracker, self.archive)
            fails, wins = loop._reflect(rows, cfg.reflect_top)

            print(f"[stage0] one-shot rewrite ({cfg.mutator_backend}/{cfg.mutator_model}) ...")
            t0 = time.time()
            cand, transcript = mutator.propose(seed, fails, wins, [], cfg.max_edits)
            tracker.log_metrics({"llm_seconds": time.time() - t0})
            open(os.path.join(run_dir, "reflection_oneshot.md"), "w").write(transcript)

            one_mv = None
            if cand is None:
                print("[stage0] one-shot produced no usable candidate.")
            else:
                cand.save(os.path.join(run_dir, "oneshot.py"))
                try:
                    cand_fn = self.sandbox.compile(cand.src)
                    self.sandbox.probe(cand_fn, probes, cfg.probe_timeout_s)
                    one_mv = self.reward.score(cand_fn, gate, src=cand.src,
                                               per_query_timeout_s=cfg.probe_timeout_s)
                    print(f"[stage0] one-shot: "
                          f"{ {k: round(one_mv.get(k), 4) for k in QUALITY_KEYS} }")
                    tracker.log_vector("oneshot_", one_mv)
                except SandboxError as e:
                    print(f"[stage0] one-shot rejected by sandbox/probe: {e}")
            tracker.log_artifacts(run_dir)

        print("\n==== stage0 decision point ====")
        print(f"seed     recall@20 = {seed_mv.get('recall@20'):.4f}")
        if one_mv:
            print(f"one-shot recall@20 = {one_mv.get('recall@20'):.4f}  "
                  "(ceiling ref ~0.381)")
        return seed_mv, one_mv

    def final_test(self, campaign):
        """The ONLY entrypoint that touches the test split."""
        cfg = self.cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        best_path = os.path.join(run_dir, "best_search.py")
        if not os.path.exists(best_path):
            raise SystemExit(f"no best_search.py in {run_dir} -- run optimize first")
        seed = self._seed_program()
        best = SearchProgram.from_file(best_path, family=seed.family)
        seed_fn = self.sandbox.compile(seed.src)
        best_fn = self.sandbox.compile(best.src)
        test_idxs = self.substrate.get_test_idxs_I_KNOW_THIS_IS_FINAL()
        print(f"[final] scoring seed AND best on the locked test split "
              f"({len(test_idxs)} queries) -- once.")

        report = {}
        with MlflowTracker(cfg).start("final-test-report", params={
                "campaign": campaign, "test_queries": len(test_idxs)}) as tracker:
            for name, fn, src in (("seed", seed_fn, seed.src), ("best", best_fn, best.src)):
                t0 = time.time()
                mv = self.reward.score(fn, test_idxs, src=src, per_query_timeout_s=30)
                report[name] = mv
                print(f"[final] {name}: { {k: round(mv.get(k), 4) for k in QUALITY_KEYS} } "
                      f"({time.time()-t0:.0f}s)")
                tracker.log_vector(f"test_{name}_", mv)

        print("\n==== FINAL TEST REPORT ====")
        for k in QUALITY_KEYS:
            d = report["best"].get(k) - report["seed"].get(k)
            print(f"{k:10s} seed {report['seed'].get(k):.4f} -> best "
                  f"{report['best'].get(k):.4f}  ({'+' if d >= 0 else ''}{d:.4f})")
        return report

    def ablate(self, strategies=("vector_only", "hybrid_rrf", "extract_first"),
               test_n=0, campaign="ablate"):
        """Seed-only attribution: score each strategy SEED on the SAME gate
        subsample (and an optional fixed test subsample) -- no optimizer, no
        Opus, embedder calls only. Answers 'is the embedder or the LLM extractor
        doing the work?' for cents. Reports B-A (sparse fusion) and C-B (LLM
        query-understanding) when the three canonical arms are present."""
        cfg = self.cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        os.makedirs(run_dir, exist_ok=True)
        gate = self.substrate.gate_idxs(run_dir, cfg.gate_size, cfg.gate_seed)
        test = None
        if test_n:
            test = sorted(random.Random(cfg.gate_seed).sample(
                self.substrate.get_test_idxs_I_KNOW_THIS_IS_FINAL(), test_n))
        probes = self._probe_queries()

        scores = {}
        for name in strategies:
            strat = load_strategy(replace(cfg, strategy=name))
            seed = SearchProgram.from_file(strat["seed"], family=strat["family"])
            fn = self.sandbox.compile(seed.src)
            self.sandbox.probe(fn, probes, cfg.probe_timeout_s)
            g = self.reward.score(fn, gate, src=seed.src,
                                  per_query_timeout_s=cfg.probe_timeout_s)
            t = (self.reward.score(fn, test, src=seed.src, per_query_timeout_s=30)
                 if test else None)
            scores[name] = (g, t)
            print(f"[ablate] {name:14s} gate { {k: round(g.get(k), 4) for k in QUALITY_KEYS} }"
                  + (f"  test { {k: round(t.get(k), 4) for k in QUALITY_KEYS} }" if t else ""))

        print("\n==== ABLATION (recall@20 / hit@1 / mrr on the gate) ====")
        for name, (g, _) in scores.items():
            print(f"{name:14s} {g.get('recall@20'):.4f} / {g.get('hit@1'):.4f} / {g.get('mrr'):.4f}")
        a, b, c = (scores.get(k, (None, None))[0]
                   for k in ("vector_only", "hybrid_rrf", "extract_first"))
        if a and b:
            print(f"B-A (sparse RRF fusion): recall@20 {b.get('recall@20') - a.get('recall@20'):+.4f}")
        if b and c:
            print(f"C-B (LLM query-understanding): recall@20 {c.get('recall@20') - b.get('recall@20'):+.4f}")
        return scores


def _git_sha():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:
        return "nogit"
