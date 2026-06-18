"""MlflowTracker -- the single place that talks to MLflow.

Mirrors the run hierarchy from the pipeline design: one parent run per
campaign, per-step metrics/artifacts under it, and an isolated final-test run.
Stage-1 logs flat metrics; Stage-2 can switch to nested child-per-candidate
runs (mlflow.dspy.autolog) without changing call sites.

Metric names are sanitized ('@' is not a legal MLflow metric character).
"""
import os

import mlflow


def mkey(name: str) -> str:
    return name.replace("@", "_at_")


def maybe_enable_openai_tracing() -> bool:
    """Phase F (OPTIONAL, gated): capture OpenAI primitive calls (G.extract /
    G.llm_rerank) as spans in MLflow's trace UI -- still "all in MLflow".

    OFF by default. Span cardinality is fine for low-volume paths (ablations,
    `final --test-n`) but would balloon storage on a full 33k-request campaign,
    so this is opt-in via MLFLOW_TRACE_OPENAI=1 and must NEVER be enabled on
    `optimize`. Returns True iff autolog was actually turned on.
    """
    if os.environ.get("MLFLOW_TRACE_OPENAI") != "1":
        return False
    try:
        mlflow.openai.autolog()
        print("[tracking] MLflow OpenAI tracing ENABLED (MLFLOW_TRACE_OPENAI=1)")
        return True
    except Exception as e:  # missing extra / version skew -- never fatal
        print(f"[tracking] OpenAI autolog unavailable: {e}")
        return False


class MlflowTracker:
    def __init__(self, cfg):
        self._cfg = cfg
        mlflow.set_tracking_uri(cfg.mlflow_url)
        mlflow.set_experiment(cfg.experiment)
        self._run = None

    def start(self, run_name, params=None):
        self._run = mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        return self

    def set_tags(self, tags: dict):
        """Run-level tags (approach/strategy/config_hash) -- what MLflow's compare
        view filters/groups on. Stringified for consistency with params."""
        mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})

    def log_metrics(self, metrics: dict, step=None):
        mlflow.log_metrics({mkey(k): float(v) for k, v in metrics.items()}, step=step)

    def log_vector(self, prefix, metric_vector, step=None):
        flat = {f"{prefix}{k}": v for k, v in metric_vector.as_flat().items()}
        self.log_metrics(flat, step=step)

    def log_artifacts(self, path):
        mlflow.log_artifacts(path)

    def log_artifact(self, path):
        mlflow.log_artifact(path)

    def end(self):
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.end()
        return False
