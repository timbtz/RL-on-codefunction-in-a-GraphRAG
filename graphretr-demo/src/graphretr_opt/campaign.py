"""Campaign -- the orchestrator. Owns all state and wires the two halves
(env + optimizer) for the three entrypoints: stage0-probe, optimize, final-test.

boot() dispatches on cfg.target: BOTH the STaRK (function) and graph_search paths
now assemble a subprocess SearchTarget stack over a FileSet candidate (the
editable service tree) -- the STaRK path over starksearch/src, the graph_search
path over graphsearch/src. Each entrypoint composes the optimizer pieces it needs
on top, driving FastLoop directly.
"""
import os
import random
import time
from dataclasses import replace

from .config import load_config, load_strategy, dump_resolved_config
from .env.errors import SandboxError
from .artifact.program import SearchProgram
from .agents.single import SingleCoder
from .agents.team import TieredCoder
from .optimizer.edit_budget import EditBudget
from .optimizer.mutator import Mutator
from .optimizer.fast_loop import FastLoop
from .tracking.mlflow_tracker import MlflowTracker, maybe_enable_openai_tracing

# NOTE: the STaRK-campaign's heavy deps (Substrate->stark_qa->torch, the
# StarkSubprocessTarget/StarkRewardAdapter, FileSet) are imported INSIDE boot()
# rather than at module top, so the graph_search path (boot_search/
# optimize_search) can be imported and run on a machine that has none of them
# installed. The STaRK path is unchanged -- it just imports a few ms later.


STARK_DOMAIN_NOTE = (
    "Target: StarkGraphSearchService over the STaRK-prime biomedical knowledge "
    "graph in FalkorDB (~129k nodes / ~8.1M directed typed edges; node label "
    "'Entity', each node has .id (int), .ntype (str), .text (str), .embedding "
    "(float[]); per-ntype vector-index labels expose ANN). The service is the "
    "WHOLE editable pipeline: it is constructed with three injected clients --\n"
    "  self._db.cypher(cypher, params) -> list[rows]   ONE read-only Cypher query.\n"
    "  self._emb.encode(text) -> list[float]           query embedding (node model).\n"
    "  self._llm(system, user, context_ids=None, model=None, max_tokens=600) -> dict\n"
    "                                                  one metered+cached JSON LLM call.\n"
    "and read-only allowlists self._db.labels / .rel_types / .ntypes. Every "
    "method (_extract, _vector_search, _text_search, _llm_rerank, _reformulate, "
    "_graph_recall, _get_neighbors, _filter_nodes) is yours to rewrite -- prompts, "
    "Cypher, fusion math, new operators. Keep `search(self, query) -> dict[int, "
    "float]` (node id -> score). Cost axes you are scored on: db_queries, "
    "llm_calls, rerank_items (LLM context volume), latency.\n"
    "HARD harness rules enforced on every self._db.cypher (a violation raises and "
    "the query is scored as a miss): READ-ONLY only (no CREATE/MERGE/SET/DELETE/"
    "REMOVE/DROP or write procs); always bound results with LIMIT; the vector "
    "`score` is cosine DISTANCE so convert to similarity (1.0 - score); "
    "algo.SPpaths must use maxLen<=3 (>=4 runs away and pegs the shared DB). "
    "Prefer algo.bfs for multi-hop recall."
)


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
        """The single bring-up entrypoint -- dispatch on `cfg.target` so ONE engine
        drives both services. `graph_search` assembles the company-KG subprocess
        stack (`boot_search`); anything else (`function`, the default) assembles
        the in-process STaRK env. The CLI sets `cfg.target` per subcommand; the
        engine reads it here instead of the caller picking boot vs boot_search."""
        if self.cfg.target == "graph_search":
            return self.boot_search()
        if self.cfg.target == "ingest_search":
            return self.boot_ingest_search()
        return self._boot_function()

    def _boot_function(self):
        """Assemble the STaRK stack -- mirrors boot_search. The candidate is now a
        whole FileSet over `starksearch/src` (the editable service file overlaid
        on the immutable base), scored in an isolated subprocess by
        StarkSubprocessTarget. NO in-process FalkorDB / RetrievalGraph / AST gate
        here: the FalkorDB + embedder + metered LLM are injected and metered in the
        worker harness, and the container-protecting caps + SPpaths/write firewall
        live in that harness's read-only DB proxy (`stark_harness.qa_runner`)."""
        cfg = self.cfg
        # The STaRK service lives in the sibling `starksearch/` package. Put the
        # monorepo root on sys.path so it resolves, exactly as boot_search does for
        # graphsearch/src.
        import sys
        if cfg.repo_root not in sys.path:
            sys.path.insert(0, cfg.repo_root)
        from starksearch.qa import Substrate
        from .env.null_sandbox import NullGraph, NullSandbox
        from .env.targets.stark_subprocess_target import StarkSubprocessTarget
        from starksearch.reward import StarkRewardAdapter
        from .artifact.file_set import FileSet

        # The function seed file is far larger than the old `search(q, G)` string,
        # so the AST complexity cap is meaningless here -- turn it OFF. The bloat
        # wall is now the value gate's TOKEN cap (set below, seed-relative), which
        # is the run10c-spiral fix; the AST axis stays a Pareto diagnostic only.
        self.cfg = cfg = replace(cfg, gate_max_complexity=0.0)

        self.substrate = Substrate(meta_holdout_size=cfg.meta_holdout_size,
                                   meta_seed=cfg.meta_seed,
                                   promote_size=getattr(cfg, "promote_size", 0),
                                   promote_seed=getattr(cfg, "promote_seed", 5678))
        target = StarkSubprocessTarget(
            falkor_cfg={"host": cfg.falkor_host, "port": cfg.falkor_port,
                        "graph_name": cfg.graph_name},
            opt_src_dir=cfg.opt_src_abs, repo_root=cfg.repo_root,
            # the worker rebuilds the harness clients from campaign.yaml+env; pass
            # root so it reads the SAME config + shares the runs_dir LLM cache.
            cfg_overrides={"root": cfg.root},
            query_concurrency=cfg.query_concurrency)
        self.reward = StarkRewardAdapter(self.substrate, target,
                                         cfg.crash_frac_limit,
                                         default_timeout_s=cfg.probe_timeout_s)
        # The candidate IS the editable service tree (string seed -> FileSet).
        self.seed = FileSet.from_base(cfg.stark_src_abs, cfg.stark_editable_files,
                                      family="stark_search")
        # Seed-from-champion (archipelago seed-chaining): overlay a saved
        # champion's primary file onto the fresh-from-base seed so the run
        # continues a prior run's best. Mirrors the final_test reload pattern
        # (campaign.final_test: seed.with_overlay({seed.primary_file: ...})).
        if getattr(cfg, "seed_champion_path", ""):
            champ = open(cfg.seed_champion_path, encoding="utf-8").read()
            self.seed = self.seed.with_overlay({self.seed.primary_file: champ})
            print(f"[campaign] seed overlaid from champion "
                  f"{cfg.seed_champion_path} -> seed sha={self.seed.sha[:8]}",
                  flush=True)
        # Enforced bloat wall: when the value gate is active, default the token cap
        # to ~1.3x the seed's program size unless the campaign set one explicitly.
        # This caps the run10c spiral (2716 -> 4607) while leaving room to grow.
        if cfg.gate_mode == "value" and not getattr(cfg, "gate_max_tokens", 0.0):
            from .reward.objectives import code_tokens
            seed_tokens = code_tokens(self.seed.src_of(self.seed.primary_file))
            self.cfg = cfg = replace(cfg, gate_max_tokens=round(seed_tokens * 1.3))
            print(f"[campaign] value gate: seed tokens={seed_tokens:.0f} -> "
                  f"max_tokens cap={cfg.gate_max_tokens:.0f}")
        self.sandbox = NullSandbox()  # no AST gate / in-process exec on this path
        self.graph = NullGraph()      # reflection get_text is inert (rows carry it)
        self.budget = None            # OpenAI metering is in the subprocess, not here
        print(f"[campaign] stark seed sha={self.seed.sha[:8]} "
              f"editable={list(self.seed.editable)} "
              f"base={cfg.stark_src_abs}")
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
        # FileSet path: a static domain note (the service API + firewall rules),
        # and safe_builtins={} (no AST gate to advertise) -- mirrors
        # _make_search_mutator. The mutator duck-types the FileSet itself.
        return Mutator(agent, STARK_DOMAIN_NOTE, safe_builtins={})

    def _seed_program(self):
        """The STaRK seed is the FileSet built in boot(); the entrypoints below
        (optimize/stage0/final/ablate) score it via `self.reward.score(..., src=
        p.src)` and FileSet.src returns the FileSet itself, so their call sites are
        unchanged."""
        return self.seed

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
        # Merge warm-start (archipelago tournament-merge): overlay each champion
        # in merge_seed_paths onto a fresh seed FileSet so the pool starts with
        # >=2 distinct champions and the loop forces COMBINE mode. Empty by
        # default => non-merge runs are byte-identical to before.
        seed_pool = []
        for p in getattr(cfg, "merge_seed_paths", ()) or ():
            src = open(p, encoding="utf-8").read()
            seed_pool.append(self.seed.with_overlay({self.seed.primary_file: src}))
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
            result = loop.run(self.substrate, seed, steps, campaign,
                              seed_pool=seed_pool)
            tracker.log_artifacts(run_dir)
        print(f"[campaign] best program -> {os.path.join(run_dir, 'best_search.py')}")
        return result

    # --------------------------------------------------- graph_search target

    def boot_search(self):
        """Assemble the graph_search stack: QaSubstrate (MCQ gold) + answerer +
        SearchTarget (fake or subprocess) + McqReward(Adapter) + FileSet seed +
        NullSandbox/NullGraph. No FalkorDB / RetrievalGraph / Sandbox / stark_qa /
        torch on this path -- the AST gate and G primitives are GONE here."""
        # The company service's qa + reward live in the sibling `graphsearch/`
        # package (carved out 2026-06-25). Put the monorepo root on sys.path so
        # `graphsearch.qa` / `graphsearch.reward` resolve, like the STaRK boot.
        import sys
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        from graphsearch.qa import QaSubstrate
        from .env.null_sandbox import NullGraph, NullSandbox
        from .env.targets.fake_target import FakeSearchTarget
        from graphsearch.reward import McqReward, StubAnswerer, McqRewardAdapter
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

    def boot_ingest_search(self):
        """Assemble the Phase-2 co-optimize stack: ONE candidate FileSet spanning
        BOTH the TS graph-ingestion code AND the Python search code, scored by the
        two-phase IngestSearchRewardAdapter (tsx ingest.ts -> Neo4j -> search.py ->
        judge-free MCQ exam). Mirrors boot_search but: (a) the seed FileSet is
        anchored at the monorepo root since its two editable files live in
        different packages (graphmod + graphsearch); (b) the reward is
        IngestSearchRewardAdapter, which owns ingest + the per-hash graph cache and
        meters total ingest+search USD; (c) the gate blends on total_usd_per_query
        (amortized) so an LLM-extraction ingest edit is admitted only if its
        accuracy gain clears its amortized build cost."""
        import sys
        if self.cfg.repo_root not in sys.path:
            sys.path.insert(0, self.cfg.repo_root)
        from graphsearch.qa import QaSubstrate
        from graphsearch.reward import McqReward, StubAnswerer
        from .env.null_sandbox import NullGraph, NullSandbox
        from .artifact.file_set import FileSet
        from .reward.ingest_search import IngestSearchRewardAdapter

        import json as _json
        n = len(_json.load(open(self.cfg.dataset_path_abs, encoding="utf-8")))
        self.cfg = self._search_cfg(self.cfg, n)
        # Phase-2: the gate must see the AMORTIZED total-pipeline cost (search +
        # ingest/N), not just search usd, so a richer-but-pricier graph is traded
        # correctly. total_usd_per_query resolves through MetricVector.get().
        from dataclasses import replace as _replace
        self.cfg = _replace(
            self.cfg,
            gate_blend=f"mcq_accuracy:1.0,total_usd_per_query:{-self.cfg.search_cost_weight}",
        )
        cfg = self.cfg

        self.search_substrate = QaSubstrate(
            dataset_path=cfg.dataset_path_abs,
            meta_holdout_size=cfg.meta_holdout_size, meta_seed=cfg.meta_seed)

        answerer = self._build_answerer()
        mcq = McqReward(answerer, crash_frac_limit=cfg.crash_frac_limit)
        # target_unused=None: the adapter builds its own in-process search target
        # (materializes search.py from the overlay, runs it over the per-hash graph).
        self.search_reward = IngestSearchRewardAdapter(
            None, mcq, self.search_substrate, cfg,
            default_timeout_s=cfg.search_timeout_s)
        # Seed spans both editable files; base = repo_root (relpaths resolve there).
        self.search_seed = FileSet.from_base(cfg.repo_root, cfg.ingest_editable_files)
        self.search_sandbox = NullSandbox()
        self.search_graph = NullGraph()
        self.budget = None
        used_stub = isinstance(answerer, StubAnswerer)
        print(f"[campaign] ingest_search seed sha={self.search_seed.sha[:8]} "
              f"editable={list(self.search_seed.editable)} "
              f"answerer={'STUB' if used_stub else cfg.answerer_model} "
              f"llm_extraction={cfg.ingest_llm_extraction}")
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
        from graphsearch.reward import StubAnswerer  # boot_search put repo_root on path
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
        from starksearch.reward import QUALITY_KEYS  # STaRK axes (boot() put it on path)
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
            fails, wins, _ = loop._reflect(rows, cfg.reflect_top)

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
        from starksearch.reward import QUALITY_KEYS  # STaRK axes (boot() put it on path)
        cfg = self.cfg
        run_dir = os.path.join(cfg.runs_dir, campaign)
        best_path = os.path.join(run_dir, "best_search.py")
        if not os.path.exists(best_path):
            raise SystemExit(f"no best_search.py in {run_dir} -- run optimize first")
        seed = self._seed_program()
        # `best_search.py` holds the accepted PRIMARY editable file (FileSet.save);
        # rebuild the best as a FileSet over the same base with that file overlaid,
        # so the subprocess target can materialize + run it like any candidate.
        best = seed.with_overlay({seed.primary_file:
                                  open(best_path, encoding="utf-8").read()})
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
        from starksearch.reward import QUALITY_KEYS  # STaRK axes (boot() put it on path)
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
