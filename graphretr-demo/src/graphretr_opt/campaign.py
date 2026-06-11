"""Campaign -- the orchestrator. Owns all state and wires the two halves
(env + optimizer) for the three entrypoints: stage0-probe, optimize, final-test.

boot() assembles the immutable env (backend -> RetrievalGraph -> Sandbox) and
the data/reward stack once; each entrypoint composes the optimizer pieces it
needs on top. Stage-1 drives FastLoop directly and ignores SlowLoop/scheduler/
momentum/gepa_adapter (all present as seams).
"""
import os
import random
import time

from .config import load_config, load_strategy
from .data.substrate import Substrate
from .env.backends.falkordb import FalkorDBBackend
from .env.cache import PrimitiveCache
from .env.embedder import QueryEmbedder
from .env.retrieval_graph import RetrievalGraph
from .env.sandbox import Sandbox, SandboxError, SAFE_BUILTINS
from .reward.evaluator import RewardModel
from .reward.objectives import QUALITY_KEYS
from .reward.pareto import ParetoArchive
from .artifact.program import SearchProgram
from .agents.single import SingleCoder
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
        self.graph = RetrievalGraph(cfg, backend, PrimitiveCache(), QueryEmbedder())
        self.sandbox = Sandbox(self.graph, default_timeout_s=cfg.probe_timeout_s)
        self.substrate = Substrate()
        self.reward = RewardModel(self.substrate, self.sandbox, cfg.crash_frac_limit)
        self.archive = ParetoArchive()
        return self

    def _make_mutator(self):
        agent = SingleCoder(self.cfg.mutator_backend, self.cfg.mutator_model,
                            self.cfg.llm_timeout_s)
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
            fails = loop._worst_failures(rows, cfg.reflect_top)

            print(f"[stage0] one-shot rewrite ({cfg.mutator_backend}/{cfg.mutator_model}) ...")
            t0 = time.time()
            cand, transcript = mutator.propose(seed, fails, [], cfg.max_edits)
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


def _git_sha():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:
        return "nogit"
