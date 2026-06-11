"""MlflowTracker -- the single place that talks to MLflow.

Mirrors the run hierarchy from the pipeline design: one parent run per
campaign, per-step metrics/artifacts under it, and an isolated final-test run.
Stage-1 logs flat metrics; Stage-2 can switch to nested child-per-candidate
runs (mlflow.dspy.autolog) without changing call sites.

Metric names are sanitized ('@' is not a legal MLflow metric character).
"""
import mlflow


def mkey(name: str) -> str:
    return name.replace("@", "_at_")


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
