"""MlflowTracker -- the single place that talks to MLflow.

Mirrors the run hierarchy from the pipeline design: one parent run per
campaign, per-step metrics/artifacts under it, and an isolated final-test run.
Stage-1 logs flat metrics; Stage-2 can switch to nested child-per-candidate
runs (mlflow.dspy.autolog) without changing call sites.

Metric names are sanitized ('@' is not a legal MLflow metric character).
"""
import contextlib
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

    @contextlib.contextmanager
    def step_span(self, name, inputs=None):
        """Phase G (OPTIONAL, gated): emit one MLflow trace per optimization
        step so the reflect -> propose -> accept/reject decision is navigable
        in the Traces tab (with the LLM transcript attached), instead of only
        living in the lineage.jsonl / reflection_*.md artifacts.

        OFF by default; opt in via MLFLOW_TRACE_STEPS=1. These are point-in-time
        spans (one per step, no per-call fan-out), so cardinality is ~steps and
        safe even on `optimize` -- but it stays gated to keep the default UI
        uncluttered. Yields the span (or None when disabled / unavailable);
        tracing must NEVER break the loop, so failures degrade to a no-op.
        """
        if os.environ.get("MLFLOW_TRACE_STEPS") != "1":
            yield None
            return
        try:
            cm = mlflow.start_span(name=name)
            span = cm.__enter__()
        except Exception as e:  # version skew / tracing backend down
            print(f"[tracking] step span unavailable: {e}")
            yield None
            return
        try:
            if inputs is not None:
                self._safe(span.set_inputs, inputs)
            yield span
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception as e:
                print(f"[tracking] step span close failed: {e}")

    def record_step(self, span, outputs=None, attributes=None):
        """Attach the step's decision payload to its span (no-op if span is
        None). Swallows errors so a tracing hiccup can never fail the step."""
        if span is None:
            return
        if attributes:
            self._safe(span.set_attributes, attributes)
        if outputs is not None:
            self._safe(span.set_outputs, outputs)

    @staticmethod
    def _safe(fn, arg):
        try:
            fn(arg)
        except Exception as e:
            print(f"[tracking] span write skipped: {e}")

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
