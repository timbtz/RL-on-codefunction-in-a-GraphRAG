"""scripts/rollup.py -- the cross-run summary the per-run MLflow views never show.

Reads every run in the experiment via MlflowClient and emits ONE page:

    run · approach · config_hash · git_sha · gate(recall@20/hit@1/mrr) ·
    held-out(recall@20) · $/run · $/accepted-edit · steps · stopped_reason

as Markdown + CSV first (a reviewable artifact, not a dashboard), then logs both
back as artifacts in a dedicated `rollup` MLflow run so the table is viewable in
the same UI. The held-out column is joined from the matching `final-test-report`
run (by the `campaign` param). `approach` is the tag that later turns scattered
runs into a three-way comparison.

Run: PYTHONPATH=$PWD/src .venv/bin/python scripts/rollup.py
     [--experiment graphretr-opt] [--out runs/rollup]
"""
import argparse
import csv
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphretr_opt.config import load_config  # noqa: E402

# Metric keys as they land in MLflow (objectives.as_flat sanitizes '@'->'_at_',
# log_vector prefixes 'best_'/'test_best_').
GATE = {"recall@20": "best_quality_recall_at_20",
        "hit@1": "best_quality_hit_at_1",
        "mrr": "best_quality_mrr"}
HELDOUT_RECALL = "test_best_quality_recall_at_20"
FINAL_RUN_NAME = "final-test-report"
ROLLUP_RUN_NAME = "rollup"


def _run_name(run):
    return run.data.tags.get("mlflow.runName") or run.info.run_name or run.info.run_id


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def _stopped_reason(m):
    if "stopped_early_at" in m:
        return f"agent_unavailable@{int(m['stopped_early_at'])}"
    if "stopped_stale_at" in m:
        return f"stale@{int(m['stopped_stale_at'])}"
    return "completed"


def collect_rows(client, experiment):
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"no MLflow experiment named {experiment!r}")
    runs = client.search_runs([exp.experiment_id], max_results=5000,
                              order_by=["start_time ASC"])

    # held-out lookup: campaign -> recall@20 on the locked test split.
    heldout = {}
    for r in runs:
        if _run_name(r) == FINAL_RUN_NAME:
            camp = r.data.params.get("campaign")
            if camp and HELDOUT_RECALL in r.data.metrics:
                heldout[camp] = r.data.metrics[HELDOUT_RECALL]

    rows = []
    for r in runs:
        name = _run_name(r)
        if name in (FINAL_RUN_NAME, ROLLUP_RUN_NAME):
            continue
        m, p, t = r.data.metrics, r.data.params, r.data.tags
        usd = m.get("usd_total", m.get("usd_cumulative"))
        acc = m.get("accepted_total")
        per_edit = (usd / acc) if (usd is not None and acc) else None
        rows.append({
            "run": name,
            "approach": t.get("approach", p.get("campaign", "-")),
            "config_hash": t.get("config_hash", p.get("config_hash", "-")),
            "git_sha": p.get("git_sha", "-"),
            "gate_recall@20": m.get(GATE["recall@20"]),
            "gate_hit@1": m.get(GATE["hit@1"]),
            "gate_mrr": m.get(GATE["mrr"]),
            "heldout_recall@20": heldout.get(p.get("campaign")),
            "usd_per_run": usd,
            "usd_per_accepted_edit": per_edit,
            "steps": int(m["steps_run"]) if "steps_run" in m else (
                int(p["steps"]) if p.get("steps", "").isdigit() else None),
            "stopped_reason": _stopped_reason(m),
        })
    return rows


COLUMNS = ["run", "approach", "config_hash", "git_sha",
           "gate_recall@20", "gate_hit@1", "gate_mrr", "heldout_recall@20",
           "usd_per_run", "usd_per_accepted_edit", "steps", "stopped_reason"]


def to_markdown(rows):
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [head, sep]
    for row in rows:
        cells = []
        for c in COLUMNS:
            v = row[c]
            if c.startswith(("gate_", "heldout_", "usd_")):
                cells.append(_fmt(v))
            elif c == "steps":
                cells.append(str(v) if v is not None else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=cfg.experiment)
    ap.add_argument("--tracking-uri", default=cfg.mlflow_url)
    ap.add_argument("--out", default=os.path.join(cfg.runs_dir, "rollup"))
    ap.add_argument("--no-log", action="store_true",
                    help="write files only; do not log a rollup run to MLflow")
    args = ap.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)
    rows = collect_rows(client, args.experiment)

    os.makedirs(args.out, exist_ok=True)
    md_path = os.path.join(args.out, "rollup.md")
    csv_path = os.path.join(args.out, "rollup.csv")
    md = to_markdown(rows)
    with open(md_path, "w") as f:
        f.write(f"# Cross-run rollup -- experiment `{args.experiment}` "
                f"({len(rows)} runs)\n\n")
        f.write(md)
    to_csv(rows, csv_path)
    print(md)
    print(f"[rollup] wrote {md_path} and {csv_path}")

    if not args.no_log:
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=ROLLUP_RUN_NAME):
            mlflow.log_metric("n_runs", len(rows))
            mlflow.log_artifact(md_path)
            mlflow.log_artifact(csv_path)
        print(f"[rollup] logged to MLflow experiment {args.experiment!r} "
              f"as run {ROLLUP_RUN_NAME!r}")
    return rows


if __name__ == "__main__":
    main()
