"""Config: one frozen dataclass; YAML overlay; env vars win last.

Precedence: dataclass defaults < configs/campaign.yaml < environment variables
(FALKOR_HOST/FALKOR_PORT/GRAPH_NAME/MLFLOW_URL/MUTATOR_BACKEND/MUTATOR_MODEL)
< explicit kwargs to load_config(). A `.env` file at the project root (KEY=VAL
lines) is read into os.environ first; if it carries ANTHROPIC_API_KEY the
mutator backend auto-switches to `sdk`.
"""
import hashlib
import os
from dataclasses import dataclass, fields

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMPAIGN_YAML = os.path.join(ROOT, "configs", "campaign.yaml")


@dataclass(frozen=True)
class Config:
    # infrastructure
    falkor_host: str = "127.0.0.1"
    falkor_port: int = 6380
    graph_name: str = "prime"
    mlflow_url: str = "http://127.0.0.1:5000"
    experiment: str = "graphretr-opt"
    root: str = ROOT

    # data / gate
    gate_size: int = 200
    gate_seed: int = 42
    gate_metric: str = "recall@20"          # axis used by 'strict' mode
    gate_mode: str = "strict"               # strict | blend | value | dominance
    gate_blend: str = "recall@20:0.6,mrr:0.4"  # weights for 'blend'/'value' Q numerator
    gate_rotate_every: int = 0              # 0 = fixed gate; N = resample gate every N steps
    gate_max_complexity: float = 0.0        # 0 = off; else candidates above this AST-complexity are ineligible
    gate_max_tokens: float = 0.0            # 0 = off; else candidates above this program-token size are ineligible (the enforced bloat wall)
    # 'value' mode: V = Q / (usd_cost^cost_exp * code_tokens^complexity_exp).
    # Exponents calibrated to break-even quality gains per resource doubling:
    #   cost x2 -> +18.75% (2^0.248), complexity x2 -> +6.25% (2^0.0875).
    # Cost penalized ~2.8x harder than complexity; both compound geometrically.
    gate_cost_exp: float = 0.248            # alpha: $/query penalty exponent
    gate_complexity_exp: float = 0.0875     # beta: code-token penalty exponent
    gate_cost_floor: float = 5e-4           # floor C so cheap-side credit can't be gamed
    gate_tokens_floor: float = 1.0          # floor K (every real program has >=1 token)

    # fast loop
    steps: int = 30
    rollout_batch: int = 24
    reflect_top: int = 8
    buffer_last: int = 8
    edit_schedule: str = "const"   # const | cosine | linear
    max_edits: int = 4             # L_max
    min_edits: int = 1             # L_min (floor for decaying schedules)
    stop_after_stale: int = 0      # 0 = run all `steps`; else stop after N stale steps
    repair_budget: int = 0         # compiler-in-the-loop self-repair: 0=off (caller compiles/probes
                                   # after propose); N => feed the sandbox/probe error back into the
                                   # SAME mutator conversation for up to N targeted fixes before rejecting

    # --- Phase 1: live-pool checkpoint/resume + graceful shutdown -----------
    checkpoint_every: int = 0      # 0 = off; else atomically snapshot runs/<c>/checkpoint.json every N steps
    resume: bool = False           # on startup, rebuild pool/incumbent/counters from checkpoint.json if present
    # --- Phase 2: process-based parallel rollout (default = exact serial) ----
    num_workers: int = 1           # 1 = serial (today's behavior, byte-identical); >1 = process pool for scoring
    rollout_fanout: int = 1        # candidates proposed+scored per step; 1 = single-candidate step (today)

    # --- run-6 optimizer evolution (Phase A/B/D) ----------------------------
    pool_enabled: bool = False     # A2/A3: instance-wise Pareto pool vs single incumbent
    pool_cap: int = 24             # A2: max pool members (frontier + top-K)
    pool_discount: bool = True     # D3: weight parent pick by 1/(1+children)
    minibatch_size: int = 0        # B2: 0=off; else cheap pre-screen size before the full gate
    minibatch_eps: float = 0.0     # 0.5: pre-screen tolerance; 0 => default 1/minibatch_size (post-mortem #4)
    meta_holdout_size: int = 0     # B3: 0=off; else val queries fenced off from the gate
    meta_seed: int = 1234          # B3: seeds the meta-holdout partition
    meta_eval_every: int = 0       # B3: 0=off; else score best on the holdout every N accepts
    # Cascaded promotion (run10c): a dedicated val slice -- disjoint from BOTH the
    # gate pool AND the meta-holdout arbiter -- on which a gate-passing candidate is
    # re-confirmed before the exported headline best moves. The cheap fixed gate
    # makes a candidate eligible; this larger slice (noise guard) decides promotion.
    promote_size: int = 0          # C: 0=off; else val queries fenced off as the promotion-confirm set
    promote_seed: int = 5678       # seeds the promotion-slice partition (distinct from meta_seed)
    promote_margin: float = 0.0    # min blend-composite gain on the promotion slice to move the exported best
    # Generation restart (run10c): when a generation stalls (stop_after_stale steps
    # with no promotion), instead of halting, restart from the Pareto set in COMBINE
    # mode (the mutator synthesizes two Pareto members). 1 = no restart = old behaviour
    # (stop on the first stall). The `steps` ceiling still bounds total work.
    max_generations: int = 1
    select_holdout_n: int = 0      # Phase 1 final bake-off: 0=use full meta-holdout; else subsample to cap cost/latency
    select_cost_floor: float = 0.0 # 0.6b: cost-aware export band; 0=pure-quality argmax. Among finalists within
                                   # this much of the top holdout value, ship the cheapest by the rerank_items meter.

    # hard caps / safety walls
    query_timeout_ms: int = 2000
    max_fanout: int = 200
    max_k: int = 3
    probe_timeout_s: float = 10.0
    crash_frac_limit: float = 0.10

    # mutator
    mutator_agent: str = "tiered"           # single | tiered (model tiering)
    mutator_backend: str = "cli"
    # explicit ids (NOT bare aliases): the CLI alias "opus" resolves to its
    # latest (4.8); we want 4.7. opus 4.7 == 4.8 price, preferred here.
    mutator_model: str = "claude-opus-4-7"      # SingleCoder model (tiered ignores it)
    analyst_model: str = "claude-haiku-4-5"     # tiered: cheap evidence digest
    editor_model: str = "claude-sonnet-4-6"     # tiered: routine edits
    architect_model: str = "claude-opus-4-7"    # tiered: plateau escalation
    architect_plateau: int = 3              # consecutive non-accepts before escalating
    llm_timeout_s: int = 900

    # openai (embedder + G.extract + G.llm_rerank); key comes from .env / OPENAI_API_KEY
    embedder: str = "minilm"                # minilm | openai_small (needs re-indexed graph)
    extract_model: str = "gpt-4o-mini"      # model behind G.extract
    rerank_model: str = "gpt-4o-mini"       # model behind G.llm_rerank
    reformulate_model: str = "gpt-4o-mini"  # model behind G.reformulate (Phase C1)
    judge_model: str = "gpt-4o-mini"        # model behind G.judge_sufficient (Item 2)
    frontier_model: str = "gpt-4o-mini"     # model behind G.pick_frontier (Item 2)
    rerank_pool_max: int = 50               # hard cap on the candidate pool sent per rerank call
    reformulate_ctx_max: int = 10           # C1: max context node docs read per reformulate call
    judge_ctx_max: int = 20                 # Item 2: max candidate docs sent per judge_sufficient call
    frontier_ctx_max: int = 20              # Item 2: max candidate docs sent per pick_frontier call
    openai_budget_usd: float = 5.0          # hard $ ceiling, persisted runs/openai_usage.json

    # strategy arm
    strategy: str = "vector_only"

    # --- STaRK (function) target: the editable FileSet service tree ----------
    # The STaRK candidate is now a whole FileSet over `starksearch/src` (the
    # editable service file overlaid on the immutable base), run in an isolated
    # subprocess exactly like graph_search -- NOT a sandboxed `search(q, G)`
    # string. Mirrors graphsearch_src / editable_files below.
    stark_src: str = "starksearch/src"     # base checkout the FileSet overlays
    stark_editable_files: tuple = (
        "stark_search/stark_graph_search_service.py",
    )                                       # relpaths the mutator may edit (tuple => hashable)

    # --- graph_search target (the real agentic search service) --------------
    target: str = "function"               # function | graph_search
    graphsearch_src: str = "graphsearch/src"          # base checkout to overlay
    dataset_path: str = "graphsearch/data/dataset.json"  # gold MCQ set
    editable_files: tuple = (
        "common/service/search/agentic_graph_traversal_search_service.py",
    )                                       # relpaths the mutator may edit (tuple => hashable)
    fake_target: bool = False              # use FakeSearchTarget (offline, no infra)
    # Neo4j creds (env: GRAPHSEARCH_NEO4J_URL/USER/PASSWORD/DATABASE)
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    # the chat model the SERVICE uses (search-time) and the MCQ ANSWERER (no judge)
    search_provider: str = "openai"
    search_model: str = "gpt-4o-mini"
    answerer_provider: str = "openai"
    answerer_model: str = "gpt-4o-mini"
    search_timeout_s: float = 1800.0       # subprocess wall-clock kill per gate BATCH
    # (one subprocess runs the gate queries, up to query_concurrency in parallel;
    # the real agentic search makes tens of LLM calls per query, so the batch needs
    # minutes, not 120s -- generous ceiling, it's a kill switch not a delay)
    eval_concurrency: int = 2              # cap concurrent candidate subprocesses (Neo4j/API)
    query_concurrency: int = 3             # graph_search: gate queries run IN PARALLEL inside
    # one worker (each is an independent, read-only search+answer -> embarrassingly
    # parallel). conc=3 -> ~90-120 peak concurrent OpenRouter calls (measured safe,
    # see graphsearch/.env). 1 = old sequential behaviour. Cuts the seed gate from
    # ~11x(search+answer) sequential to ceil(11/conc) waves.
    # Cost-aware gate (graph_search): the gate admits on a blended composite
    #   composite = mcq_accuracy - search_cost_weight * usd_cost
    # where usd_cost is the REAL mean OpenRouter $/query (search + answerer). This
    # is THE knob that makes cost crucial: raise it to push the optimizer toward
    # cheaper architectures (batched relevance call, BM25/no-LLM pruning, leaner
    # prompts); lower it to prioritize accuracy. Tune after seeing the seed's cost
    # (keep seed composite > 0: weight < seed_accuracy / seed_usd_cost).
    search_cost_weight: float = 5.0

    @property
    def runs_dir(self):
        return os.path.join(self.root, "runs")

    @property
    def repo_root(self):
        """The monorepo root (parent of graphretr-demo) -- graphsearch is a
        sibling of the optimizer, so graphsearch_src / dataset_path resolve
        against this, not the optimizer's own root."""
        return os.path.dirname(self.root)

    def _resolve(self, path):
        return path if os.path.isabs(path) else os.path.join(self.repo_root, path)

    @property
    def stark_src_abs(self):
        """The STaRK service base tree (`starksearch/src`) the FileSet overlays,
        resolved against the monorepo root (sibling of graphretr-demo)."""
        return self._resolve(self.stark_src)

    @property
    def graphsearch_src_abs(self):
        return self._resolve(self.graphsearch_src)

    @property
    def dataset_path_abs(self):
        return self._resolve(self.dataset_path)

    @property
    def opt_src_abs(self):
        """The optimizer's own `src` dir (put on the worker subprocess's
        PYTHONPATH so `python -m graphretr_opt..._worker` resolves)."""
        return os.path.join(self.root, "src")

    # ---- resolved-config snapshot + hash (auditability, Phase A) -----------
    # The full effective config AFTER defaults < campaign.yaml < env < kwargs is
    # resolved. `root` is excluded: it is a machine-specific absolute path and
    # would make the hash non-portable (the same run on another checkout would
    # otherwise look like a different experiment).
    def resolved_dict(self) -> dict:
        """All resolved fields except the machine-specific `root`, key-sorted."""
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "root"}
        return dict(sorted(d.items()))

    def resolved_yaml(self) -> str:
        """Deterministic YAML dump of the effective config (stable key order)."""
        return yaml.safe_dump(self.resolved_dict(), sort_keys=True,
                              default_flow_style=False)

    def config_hash(self) -> str:
        """12-hex content hash of `resolved_yaml()`. Same hash => same experiment;
        a changed hash under an unchanged git_sha flags silent env/CLI drift."""
        return hashlib.sha256(self.resolved_yaml().encode()).hexdigest()[:12]


def _load_dotenv():
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(**overrides) -> Config:
    _load_dotenv()
    kw = {}
    if os.path.exists(CAMPAIGN_YAML):
        doc = yaml.safe_load(open(CAMPAIGN_YAML)) or {}
        known = {f.name for f in fields(Config)}
        unknown = set(doc) - known
        if unknown:
            raise ValueError(f"unknown keys in campaign.yaml: {sorted(unknown)}")
        kw.update(doc)
    env = {
        "falkor_host": os.environ.get("FALKOR_HOST"),
        "falkor_port": int(os.environ["FALKOR_PORT"]) if os.environ.get("FALKOR_PORT") else None,
        "graph_name": os.environ.get("GRAPH_NAME"),
        "mlflow_url": os.environ.get("MLFLOW_URL"),
        "mutator_backend": os.environ.get("MUTATOR_BACKEND"),
        "mutator_model": os.environ.get("MUTATOR_MODEL"),
        # graph_search target (own GRAPHSEARCH_* namespace, documented in Task 0)
        "neo4j_url": os.environ.get("GRAPHSEARCH_NEO4J_URL"),
        "neo4j_user": os.environ.get("GRAPHSEARCH_NEO4J_USER"),
        "neo4j_password": os.environ.get("GRAPHSEARCH_NEO4J_PASSWORD"),
        "neo4j_database": os.environ.get("GRAPHSEARCH_NEO4J_DATABASE"),
        "answerer_model": os.environ.get("ANSWERER_MODEL"),
        "search_model": os.environ.get("SEARCH_MODEL"),
        "search_provider": os.environ.get("SEARCH_PROVIDER"),
        "answerer_provider": os.environ.get("ANSWERER_PROVIDER"),
        "query_concurrency": (int(os.environ["QUERY_CONCURRENCY"])
                              if os.environ.get("QUERY_CONCURRENCY") else None),
        "search_cost_weight": (float(os.environ["SEARCH_COST_WEIGHT"])
                               if os.environ.get("SEARCH_COST_WEIGHT") else None),
        "fake_target": True if os.environ.get("GRAPHRETR_FAKE_TARGET") == "1" else None,
    }
    if env["mutator_backend"] is None and os.environ.get("ANTHROPIC_API_KEY"):
        env["mutator_backend"] = "sdk"
    kw.update({k: v for k, v in env.items() if v is not None})
    kw.update(overrides)
    if isinstance(kw.get("editable_files"), list):  # yaml -> tuple (hashable)
        kw["editable_files"] = tuple(kw["editable_files"])
    if isinstance(kw.get("stark_editable_files"), list):  # yaml -> tuple (hashable)
        kw["stark_editable_files"] = tuple(kw["stark_editable_files"])
    return Config(**kw)


def dump_resolved_config(cfg: Config, run_dir: str):
    """Write the effective config to <run_dir>/resolved_config.yaml and return
    (path, config_hash). The on-disk YAML is byte-identical to what config_hash()
    hashes, so a reviewer can recompute the hash from the artifact alone."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "resolved_config.yaml")
    with open(path, "w") as f:
        f.write(cfg.resolved_yaml())
    return path, cfg.config_hash()


def load_strategy(cfg: Config) -> dict:
    """configs/strategies/<name>.yaml -> {'family', 'seed' (abs path), ...}."""
    path = os.path.join(cfg.root, "configs", "strategies", f"{cfg.strategy}.yaml")
    doc = yaml.safe_load(open(path))
    doc["seed"] = os.path.join(cfg.root, doc["seed"])
    return doc
