"""Config: one frozen dataclass; YAML overlay; env vars win last.

Precedence: dataclass defaults < configs/campaign.yaml < environment variables
(FALKOR_HOST/FALKOR_PORT/GRAPH_NAME/MLFLOW_URL/MUTATOR_BACKEND/MUTATOR_MODEL)
< explicit kwargs to load_config(). A `.env` file at the project root (KEY=VAL
lines) is read into os.environ first; if it carries ANTHROPIC_API_KEY the
mutator backend auto-switches to `sdk`.
"""
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
    gate_metric: str = "recall@20"

    # fast loop
    steps: int = 30
    rollout_batch: int = 24
    reflect_top: int = 8
    buffer_last: int = 8
    edit_schedule: str = "const"   # const | cosine | linear
    max_edits: int = 4             # L_max
    min_edits: int = 1             # L_min (floor for decaying schedules)

    # hard caps / safety walls
    query_timeout_ms: int = 2000
    max_fanout: int = 200
    max_k: int = 3
    probe_timeout_s: float = 10.0
    crash_frac_limit: float = 0.10

    # mutator
    mutator_backend: str = "cli"
    mutator_model: str = "opus"
    llm_timeout_s: int = 900

    # strategy arm
    strategy: str = "vector_only"

    @property
    def runs_dir(self):
        return os.path.join(self.root, "runs")


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


def load_strategy(cfg: Config) -> dict:
    """configs/strategies/<name>.yaml -> {'family', 'seed' (abs path), ...}."""
    path = os.path.join(cfg.root, "configs", "strategies", f"{cfg.strategy}.yaml")
    doc = yaml.safe_load(open(path))
    doc["seed"] = os.path.join(cfg.root, doc["seed"])
    return doc
