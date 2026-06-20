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
    gate_mode: str = "strict"               # strict | blend | dominance
    gate_blend: str = "recall@20:0.6,mrr:0.4"  # weights for 'blend' mode
    gate_rotate_every: int = 0              # 0 = fixed gate; N = resample gate every N steps
    gate_max_complexity: float = 0.0        # 0 = off; else candidates above this AST-complexity are ineligible

    # fast loop
    steps: int = 30
    rollout_batch: int = 24
    reflect_top: int = 8
    buffer_last: int = 8
    edit_schedule: str = "const"   # const | cosine | linear
    max_edits: int = 4             # L_max
    min_edits: int = 1             # L_min (floor for decaying schedules)
    stop_after_stale: int = 0      # 0 = run all `steps`; else stop after N stale steps

    # --- run-6 optimizer evolution (Phase A/B/D) ----------------------------
    pool_enabled: bool = False     # A2/A3: instance-wise Pareto pool vs single incumbent
    pool_cap: int = 24             # A2: max pool members (frontier + top-K)
    pool_discount: bool = True     # D3: weight parent pick by 1/(1+children)
    minibatch_size: int = 0        # B2: 0=off; else cheap pre-screen size before the full gate
    minibatch_eps: float = 0.0     # 0.5: pre-screen tolerance; 0 => default 1/minibatch_size (post-mortem #4)
    meta_holdout_size: int = 0     # B3: 0=off; else val queries fenced off from the gate
    meta_seed: int = 1234          # B3: seeds the meta-holdout partition
    meta_eval_every: int = 0       # B3: 0=off; else score best on the holdout every N accepts
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
    rerank_pool_max: int = 50               # hard cap on the candidate pool sent per rerank call
    reformulate_ctx_max: int = 10           # C1: max context node docs read per reformulate call
    openai_budget_usd: float = 5.0          # hard $ ceiling, persisted runs/openai_usage.json

    # strategy arm
    strategy: str = "vector_only"

    @property
    def runs_dir(self):
        return os.path.join(self.root, "runs")

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
    }
    if env["mutator_backend"] is None and os.environ.get("ANTHROPIC_API_KEY"):
        env["mutator_backend"] = "sdk"
    kw.update({k: v for k, v in env.items() if v is not None})
    kw.update(overrides)
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
