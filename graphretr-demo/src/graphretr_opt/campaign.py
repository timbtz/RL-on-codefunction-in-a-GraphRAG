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

from .config import load_config, load_strategy, dump_resolved_config
from .env.sandbox import SandboxError, SAFE_BUILTINS
from .reward.objectives import QUALITY_KEYS
from .artifact.program import SearchProgram
from .agents.single import SingleCoder
from .agents.team import TieredCoder
from .optimizer.edit_budget import EditBudget
from .optimizer.mutator import Mutator
from .optimizer.fast_loop import FastLoop
from .tracking.mlflow_tracker import MlflowTracker, maybe_enable_openai_tracing

# NOTE: the function-campaign's heavy deps (Substrate->stark_qa, RewardModel->
# torch, FalkorDBBackend->falkordb, RetrievalGraph/embedder->numpy) are imported
# INSIDE boot() rather than at module top, so the graph_search path
# (boot_search/optimize_search) can be imported and run on a machine that has
# none of them installed. The function path is unchanged -- it just imports a
# few ms later.


SEARCH_DOMAIN_NOTE = (
    "Target: AgenticGraphTraversalSearchService over a Neo4j knowledge graph "
    "(German company KG). Retrieval = fulltext seed lookup on the `ft_Entities` "
    "index, then recursive neighbour traversal; each frontier node's Documents "
    "are LLM-filtered for usefulness; an LLM judges sufficiency and picks which "
    "entities to expand next. Cost levers (also the cost axes you are scored on): "
    "max_depth, neighbor_limit, docs_per_entity, max_frontier, max_workers, and "
    "how many LLM calls per node (_filter_useful_docs / _judge_sufficient / "
    "_pick_next_entities). Documents carry [doc:ID] citations. There are NO "
    "embeddings. Keep `search(query) -> str` returning the retrieved context."
)


class Campaign:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()

    # --------------------------------------------------------------- bring-up

    def boot(self):
        cfg = self.cfg
        # function-campaign deps imported lazily (see module-top note)
        from .data.substrate import Substrate
        from .env.backends.falkordb import FalkorDBBackend
        from .env.cache import PrimitiveCache
        from .env.embedder import make_embedder
        from .env.openai_client import OpenAIBudget
        from .env.retrieval_graph import RetrievalGraph
        from .env.sandbox import Sandbox
        from .reward.evaluator import RewardModel
        backend = FalkorDBBackend(cfg.falkor_host, cfg.falkor_port, cfg.graph_name)
        # One shared metered OpenAI gateway for the embedder AND G.extract;
        # without a key both stay disabled and the minilm path works as before.
        budget = None
        if os.environ.get("OPENAI_API_KEY"):
            budget = OpenAIBudget(os.path.join(cfg.runs_dir, "openai_usage.json"),
                                  ceiling_usd=cfg.openai_budget_usd)
            print(f"[campaign] openai budget: ${budget.spent_usd:.2f} spent of "
                  f"${cfg.openai_budget_usd:.2f} ceiling (embedder={cfg.embedder})")
        self.budget = budget  # shared OpenAI gateway; FastLoop reads it for per-step cost
        self.graph = RetrievalGraph(cfg, backend, PrimitiveCache(),
                                    make_embedder(cfg, budget), llm_budget=budget)
        self.sandbox = Sandbox(self.graph, default_timeout_s=cfg.probe_timeout_s)
        self.substrate = Substrate(meta_holdout_size=cfg.meta_holdout_size,
                                   meta_seed=cfg.meta_seed)
        self.reward = RewardModel(self.substrate, self.sandbox, cfg.crash_frac_limit)
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
        cfg_path, cfg_hash = dump_resolved_config(cfg, run_dir)

        with MlflowTracker(cfg).start(campaign, params={
            **{k: v for k, v in vars(cfg).items() if k != "root"},
            "campaign": campaign, "steps": steps, "git_sha": _git_sha(),
            "config_hash": cfg_hash,
        }) as tracker:
            tracker.set_tags({"approach": campaign, "strategy": cfg.strategy,
                              "config_hash": cfg_hash})
            loop = FastLoop(cfg, self.graph, self.sandbox, self.reward, mutator,
                            edit_budget, tracker, budget=self.budget)
            result = loop.run(self.substrate, seed, steps, campaign)
            tracker.log_artifacts(run_dir)
        print(f"[campaign] best program -> {os.path.join(run_dir, 'best_search.py')}")
        return result

    # --------------------------------------------------- graph_search target

    def boot_search(self):
        """Assemble the graph_search stack: QaSubstrate (MCQ gold) + answerer +
        SearchTarget (fake or subprocess) + McqReward(Adapter) + FileSet seed +
        NullSandbox/NullGraph. No FalkorDB / RetrievalGraph / Sandbox / stark_qa /
        torch on this path -- the AST gate and G primitives are GONE here."""
        from .data.qa_substrate import QaSubstrate
        from .env.null_sandbox import NullGraph, NullSandbox
        from .env.targets.fake_target import FakeSearchTarget
        from .reward.mcq_reward import McqReward, StubAnswerer
        from .reward.mcq_reward_adapter import McqRewardAdapter
        from .artifact.file_set import FileSet
        # campaign.yaml is tuned for STaRK (gate_size=1941, recall@20 gate,
        # complexity cap, large meta-holdout). The MCQ set is tiny and the metric
        # is mcq_accuracy, so re-derive the gate knobs for this target from the
        # dataset size before building anything (explicit env/kwargs still win
        # where they make sense).
        import json as _json
        n = len(_json.load(open(self.cfg.dataset_path_abs, encoding="utf-8")))
        self.cfg = self._search_cfg(self.cfg, n)
        cfg = self.cfg

        self.search_substrate = QaSubstrate(
            dataset_path=cfg.dataset_path_abs,
            meta_holdout_size=cfg.meta_holdout_size, meta_seed=cfg.meta_seed)

        answerer = self._build_answerer()
        if cfg.fake_target:
            target = FakeSearchTarget()
            print("[campaign] graph_search: FakeSearchTarget (offline, no infra)")
        else:
            from .env.targets.subprocess_target import SubprocessSearchTarget
            # CLI-only runs: route in-search calls through the Claude CLI and
            # SHRINK the search's per-eval call volume so an eval can finish within
            # subscription rate limits (the full-fidelity ~60 calls/query would hit
            # limits mid-eval). Fewer frontier nodes/depth/workers = far fewer CLI
            # calls; also lower max_workers so we don't fire 20 concurrent `claude`.
            svc_kwargs = {}
            if cfg.search_provider.lower() in ("claude_cli", "cli"):
                svc_kwargs = {"max_frontier": 4, "max_depth": 1,
                              "docs_per_entity": 8, "neighbor_limit": 10,
                              "max_workers": 3}
            target = SubprocessSearchTarget(
                neo4j_cfg={"url": cfg.neo4j_url, "username": cfg.neo4j_user,
                           "password": cfg.neo4j_password,
                           "database": cfg.neo4j_database},
                llm_cfg={"provider": cfg.search_provider, "model": cfg.search_model},
                opt_src_dir=cfg.opt_src_abs,
                service_relpath=cfg.editable_files[0] if cfg.editable_files else None,
                service_kwargs=svc_kwargs or None,
                query_concurrency=cfg.query_concurrency)
            print(f"[campaign] graph_search: SubprocessSearchTarget over "
                  f"{cfg.graphsearch_src_abs} (provider={cfg.search_provider}, "
                  f"query_concurrency={cfg.query_concurrency}, svc_kwargs={svc_kwargs})")

        mcq = McqReward(answerer, crash_frac_limit=cfg.crash_frac_limit)
        self.search_reward = McqRewardAdapter(
            target, mcq, self.search_substrate,
            default_timeout_s=cfg.search_timeout_s)
        self.search_seed = FileSet.from_base(cfg.graphsearch_src_abs,
                                             cfg.editable_files)
        self.search_sandbox = NullSandbox()
        self.search_graph = NullGraph()
        self.budget = None  # OpenAI budget metering is in the subprocess, not here
        used_stub = isinstance(answerer, StubAnswerer)
        print(f"[campaign] graph_search seed sha={self.search_seed.sha[:8]} "
              f"editable={list(self.search_seed.editable)} "
              f"answerer={'STUB' if used_stub else cfg.answerer_model}")
        return self

    @staticmethod
    def _search_cfg(cfg, n):
        """Re-derive STaRK-specific gate knobs for the graph_search target from
        the MCQ dataset size `n`. Switches the gate to mcq_accuracy/strict, drops
        the AST complexity cap (the real service file is far larger than the
        function seed -- the cap is meaningless here), and shrinks gate /
        minibatch / meta-holdout so the small set yields a non-empty gate pool."""
        # meta-holdout must leave a non-empty gate pool; disable on tiny sets
        mh = cfg.meta_holdout_size if (cfg.meta_holdout_size
                                       and cfg.meta_holdout_size < n) else 0
        if mh and (n - mh) < 1:
            mh = 0
        # Cheap pre-screen: score a candidate on a small REPRESENTATIVE subsample
        # first; only candidates that hold up pay for the full gate. Representative
        # (not failures-only) so the screen stays cost-aware AND catches
        # regressions on previously-passing questions -- a failures-only screen
        # would wrongly kill cheaper-same-accuracy wins and miss breakage.
        # MUST be well below the full gate (~half): a candidate that PASSES the
        # screen then pays the full gate too, so b≈gate would add cost, not save
        # it. Ignore the STaRK default (large); size from the MCQ set instead.
        mb = max(1, n // 2)
        return replace(
            cfg,
            gate_metric="mcq_accuracy",
            # Cost-aware gate: admit on composite = mcq_accuracy - w*usd_cost, so
            # the optimizer actively trades REAL dollars against accuracy (the
            # whole point -- raw call count was the wrong unit). w is tunable via
            # search_cost_weight / SEARCH_COST_WEIGHT. usd_cost resolves through
            # MetricVector.get()'s attr fallback.
            gate_mode="blend",
            gate_blend=f"mcq_accuracy:1.0,usd_cost:{-cfg.search_cost_weight}",
            gate_max_complexity=0.0,
            gate_rotate_every=0,
            gate_size=min(cfg.gate_size, n),
            rollout_batch=min(cfg.rollout_batch, n),  # train pool is the tiny MCQ set
            meta_holdout_size=mh,
            minibatch_size=mb,
            meta_eval_every=(cfg.meta_eval_every if mh else 0),
            # FastLoop passes `probe_timeout_s` as the per-eval wall-clock; the
            # STaRK default (10s) suits sub-second in-process probes, but here it
            # is the timeout for a whole subprocess batch of real agentic queries
            # (tens of LLM calls each). Use the graph_search batch budget instead.
            probe_timeout_s=cfg.search_timeout_s,
        )

    def _build_answerer(self):
        """The MCQ answerer (true-external dep; temperature=0). Falls back to the
        deterministic StubAnswerer when running the fake target with no API key
        (the offline smoke path)."""
        from .reward.mcq_reward import StubAnswerer
        cfg = self.cfg
        has_key = bool(os.environ.get("OPENAI_API_KEY")
                       or os.environ.get("ANTHROPIC_API_KEY")
                       or os.environ.get("OPENROUTER_API_KEY"))
        if cfg.fake_target and not has_key:
            return StubAnswerer()
        provider = cfg.answerer_provider.lower()
        if provider in ("claude_cli", "cli"):
            # Route the MCQ answerer through the Claude CLI (subscription) too.
            import sys
            if cfg.graphsearch_src_abs not in sys.path:
                sys.path.insert(0, cfg.graphsearch_src_abs)
            from common.service.qa_eval.qa_runner import ClaudeCliChat
            return ClaudeCliChat(model=cfg.answerer_model)
        if provider in ("openai", "azure_openai"):
            from langchain_openai import ChatOpenAI
            # Route through OpenRouter (OpenAI-compatible) when configured.
            extra = {}
            if os.environ.get("OPENROUTER_API_KEY"):
                extra = {"base_url": os.environ.get("OPENROUTER_BASE_URL",
                                                    "https://openrouter.ai/api/v1"),
                         "api_key": os.environ["OPENROUTER_API_KEY"],
                         # usage.include -> real USD cost in token_usage.cost,
                         # read by mcq_reward._answerer_cost
                         "extra_body": {"usage": {"include": True}}}
            return ChatOpenAI(model=cfg.answerer_model, temperature=0, **extra)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=cfg.answerer_model, temperature=0)
        raise ValueError(f"unknown answerer provider {provider!r}")

    def _make_search_mutator(self):
        cfg = self.cfg
        if cfg.mutator_agent == "tiered":
            agent = TieredCoder(cfg.mutator_backend, cfg.analyst_model,
                                cfg.editor_model, cfg.architect_model,
                                cfg.llm_timeout_s)
        else:
            agent = SingleCoder(cfg.mutator_backend, cfg.mutator_model,
                                cfg.llm_timeout_s)
        return Mutator(agent, SEARCH_DOMAIN_NOTE, safe_builtins={})

    def optimize_search(self, steps=None, campaign="search"):
        """The graph_search entrypoint -- mirrors optimize() but wires the
        SearchTarget + McqReward stack into FastLoop with a NullSandbox (no AST
        gate / in-process exec; isolation is the subprocess). Logs the MCQ dataset
        + per-question recap tables to MLflow."""
        cfg = self.cfg
        steps = steps if steps is not None else cfg.steps
        edit_budget = EditBudget(cfg.edit_schedule, cfg.max_edits, cfg.min_edits,
                                 max(1, steps))
        mutator = self._make_search_mutator()
        seed = self.search_seed
        run_dir = os.path.join(cfg.runs_dir, campaign)
        os.makedirs(run_dir, exist_ok=True)
        cfg_path, cfg_hash = dump_resolved_config(cfg, run_dir)

        with MlflowTracker(cfg).start(campaign, params={
            **{k: v for k, v in vars(cfg).items() if k != "root"},
            "campaign": campaign, "steps": steps, "git_sha": _git_sha(),
            "config_hash": cfg_hash, "target": "graph_search",
        }) as tracker:
            tracker.set_tags({"approach": campaign, "target": "graph_search",
                              "config_hash": cfg_hash})
            tracker.log_dataset(cfg.dataset_path_abs, context="eval")
            loop = FastLoop(cfg, self.search_graph, self.search_sandbox,
                            self.search_reward, mutator, edit_budget, tracker,
                            budget=None)
            result = loop.run(self.search_substrate, seed, steps, campaign)
            self._log_search_recap(tracker, run_dir, result.best_program)
            tracker.log_artifacts(run_dir)
        mv = result.best_metrics
        print(f"[campaign] graph_search best quality: "
              f"{ {k: round(v, 4) for k, v in (mv.quality.items() if mv else [])} }")
        print(f"[campaign] best overlay -> {os.path.join(run_dir, 'best_search.py')}")
        return result

    def _log_search_recap(self, tracker, run_dir, best_program):
        """Score the FINAL best FileSet on the gate with per-question rows and log
        them as an MLflow recap table (chosen/gold/correct/retrieval_hit) -- the
        fix for "we see virtually nothing in MLflow". Also persisted as a Parquet/
        JSON artifact for offline analysis."""
        sub = self.search_substrate
        gate = sub.gate_idxs(run_dir, self.cfg.gate_size, self.cfg.gate_seed)
        try:
            _, rows = self.search_reward.score(
                best_program, gate, return_rows=True,
                per_query_timeout_s=self.cfg.search_timeout_s)
        except Exception as e:
            print(f"[campaign] recap scoring skipped ({type(e).__name__}: {e})")
            return
        recap = sub.eval_log_rows(rows, candidate_sha=best_program.sha[:8])
        tracker.log_table(recap, artifact_file="recap/best_recap.json")
        import json as _json
        path = os.path.join(run_dir, "best_recap.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(recap, f, ensure_ascii=False, indent=1)
        n_hit = sum(r["retrieval_hit"] for r in recap)
        n_ob = sum(r["openbook_correct"] for r in recap)
        print(f"[campaign] recap: {len(recap)} questions, retrieval_hit={n_hit}, "
              f"openbook_correct={n_ob} -> {path}")

    def stage0(self, campaign="stage0"):
        """Headroom check: score the seed, then ONE one-shot rewrite (no loop)."""
        cfg = self.cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        os.makedirs(run_dir, exist_ok=True)
        cfg_path, cfg_hash = dump_resolved_config(cfg, run_dir)
        gate = self.substrate.gate_idxs(run_dir, cfg.gate_size, cfg.gate_seed)
        probes = self._probe_queries()
        seed = self._seed_program()
        seed_fn = self.sandbox.compile(seed.src)
        self.sandbox.probe(seed_fn, probes, cfg.probe_timeout_s)
        mutator = self._make_mutator()

        with MlflowTracker(cfg).start(campaign, params={
                **{k: v for k, v in vars(cfg).items() if k != "root"},
                "config_hash": cfg_hash}) as tracker:
            tracker.set_tags({"approach": campaign, "strategy": cfg.strategy,
                              "config_hash": cfg_hash})
            maybe_enable_openai_tracing()  # Phase F: gated, low-volume path only
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
                            EditBudget("const", cfg.max_edits), tracker)
            fails, wins = loop._reflect(rows, cfg.reflect_top)

            print(f"[stage0] one-shot rewrite ({cfg.mutator_backend}/{cfg.mutator_model}) ...")
            t0 = time.time()
            cand, transcript, _ = mutator.propose(seed, fails, wins, [], cfg.max_edits)
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

    def final_test(self, campaign, test_n=0):
        """The ONLY entrypoint that touches the test split. With test_n>0 scores a
        deterministic test subsample (same seed family as ablate) -- a cheap honest
        verdict that keeps the LLM-rerank bill bounded; test_n=0 = full 2801 split."""
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
        if test_n:
            test_idxs = sorted(random.Random(cfg.gate_seed).sample(test_idxs, test_n))
        print(f"[final] scoring seed AND best on the locked test split "
              f"({len(test_idxs)} queries{' SUBSAMPLE' if test_n else ''}) -- once.")

        cfg_hash = cfg.config_hash()
        report = {}
        with MlflowTracker(cfg).start("final-test-report", params={
                "campaign": campaign, "test_queries": len(test_idxs),
                "config_hash": cfg_hash}) as tracker:
            tracker.set_tags({"approach": campaign, "strategy": cfg.strategy,
                              "config_hash": cfg_hash})
            maybe_enable_openai_tracing()  # Phase F: gated, low-volume path only
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
