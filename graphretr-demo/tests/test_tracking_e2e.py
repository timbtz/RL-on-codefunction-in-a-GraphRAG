"""End-to-end wiring test for the tracking plan (verification steps 3-4).

Codifies the plan's "assert in the MLflow store, scripted, not by eye" check so
the wiring stays proven on every future run. It is HERMETIC: it logs a 2-step
mini-run the way campaign.py/fast_loop.py do (config_hash param + resolved
config artifact, per-step cost metrics, approach/strategy/config_hash tags,
lineage.jsonl artifact) into a temporary `file://` MLflow store, then reads it
back via MlflowClient and runs scripts/rollup.py over the same store -- no live
server, no FalkorDB, no network.

A separate live smoke (README): `cli optimize --campaign-name trk_smoke
--steps 2` against the running :5000 server exercises the same fields end-to-end.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_tracking_e2e
"""
import hashlib
import importlib.util
import json
import os
import tempfile

# MLflow 3.x puts the local file store in maintenance mode behind this opt-out;
# the hermetic test uses a temp file:// store (the live server runs on sqlite).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from graphretr_opt.config import load_config, dump_resolved_config  # noqa: E402
from graphretr_opt.tracking.mlflow_tracker import MlflowTracker  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _load_rollup():
    spec = importlib.util.spec_from_file_location(
        "rollup", os.path.join(REPO, "scripts", "rollup.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_lineage(run_dir, shas):
    path = os.path.join(run_dir, "lineage.jsonl")
    with open(path, "w") as f:
        for i in range(len(shas) - 1):
            f.write(json.dumps({
                "step": i, "parent_sha": shas[i], "child_sha": shas[i + 1],
                "change_summary": f"edit {i}", "accepted": True,
                "reason": "accepted (gate improved)",
                "metric_vector": {"quality_recall_at_20": 0.30 + 0.01 * i},
                "tokens_step": 10, "edit_budget": 2, "gate_tag": "fix",
            }) + "\n")
    return path


def test_e2e_wiring():
    with tempfile.TemporaryDirectory() as d:
        uri = "file://" + os.path.join(d, "mlruns")
        cfg = load_config(root=d, mlflow_url=uri, experiment="trk_test")
        run_dir = os.path.join(d, "runs", "trk_smoke")

        cfg_path, cfg_hash = dump_resolved_config(cfg, run_dir)
        shas = ["a" * 64, "b" * 64, "c" * 64]      # 2-step parent->child chain
        lineage_path = _write_lineage(run_dir, shas)

        run_id = None
        with MlflowTracker(cfg).start("trk_smoke", params={
                "campaign": "trk_smoke", "steps": 2, "git_sha": "deadbeef",
                "config_hash": cfg_hash}) as tracker:
            run_id = tracker._run.info.run_id
            tracker.set_tags({"approach": "trk_smoke", "strategy": cfg.strategy,
                              "config_hash": cfg_hash})
            for i in range(2):
                tracker.log_metrics({
                    "tokens_accepted": 10, "tokens_rejected": 0,
                    "usd_step": 0.10, "usd_cumulative": 0.10 * (i + 1),
                    "best_quality_recall_at_20": 0.30 + 0.01 * i,
                    "best_quality_hit_at_1": 0.10, "best_quality_mrr": 0.10,
                }, step=i)
            tracker.log_metrics({"accepted_total": 2, "steps_run": 2, "usd_total": 0.20})
            tracker.log_artifact(cfg_path)
            tracker.log_artifact(lineage_path)

        client = MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)

        # --- config_hash param present AND equals the dumped artifact's hash ---
        disk_hash = hashlib.sha256(open(cfg_path).read().encode()).hexdigest()[:12]
        _check("param config_hash present", "config_hash" in run.data.params)
        _check("param config_hash == hash(resolved_config.yaml)",
               run.data.params["config_hash"] == cfg_hash == disk_hash)

        # --- per-step cost metrics exist for each step --------------------------
        for key in ("tokens_accepted", "tokens_rejected", "usd_cumulative"):
            hist = client.get_metric_history(run_id, key)
            _check(f"per-step metric {key} has a point per step",
                   sorted(m.step for m in hist) == [0, 1])

        # --- tags set -----------------------------------------------------------
        _check("tag approach set", run.data.tags.get("approach") == "trk_smoke")
        _check("tag strategy set", run.data.tags.get("strategy") == cfg.strategy)
        _check("tag config_hash set", run.data.tags.get("config_hash") == cfg_hash)

        # --- artifacts attached, and lineage chains parent->child ---------------
        names = {a.path for a in client.list_artifacts(run_id)}
        _check("resolved_config.yaml attached", "resolved_config.yaml" in names)
        _check("lineage.jsonl attached", "lineage.jsonl" in names)

        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="lineage.jsonl",
            dst_path=os.path.join(d, "dl"), tracking_uri=uri)
        rows = [json.loads(l) for l in open(local) if l.strip()]
        _check("lineage has one row per step", len(rows) == 2)
        _check("lineage parent->child chain intact",
               rows[0]["child_sha"] == rows[1]["parent_sha"])

        # --- rollup includes trk_smoke with correct numbers (step 4) ------------
        rollup = _load_rollup()
        summary = {r["run"]: r for r in rollup.collect_rows(client, "trk_test")}
        _check("rollup includes trk_smoke", "trk_smoke" in summary)
        row = summary["trk_smoke"]
        _check("rollup config_hash", row["config_hash"] == cfg_hash)
        _check("rollup approach", row["approach"] == "trk_smoke")
        _check("rollup gate recall@20", abs(row["gate_recall@20"] - 0.31) < 1e-9)
        _check("rollup $/run", abs(row["usd_per_run"] - 0.20) < 1e-9)
        _check("rollup $/accepted-edit (usd_total / accepted_total)",
               abs(row["usd_per_accepted_edit"] - 0.10) < 1e-9)
        _check("rollup steps", row["steps"] == 2)


def main():
    test_e2e_wiring()
    print("\nall tracking e2e tests passed")


if __name__ == "__main__":
    main()
