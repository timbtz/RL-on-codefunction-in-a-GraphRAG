"""Phase 3: read-only lineage visualizer + static HTML export.

Builds a synthetic runs/<campaign>/ (lineage.jsonl + step_/reflection_ side files,
including a row with a non-finite metric) and asserts:
  * the parent->child graph is reconstructed (seed root, accepted/rejected nodes,
    no-candidate rows folded onto their parent),
  * inf/NaN metrics are sanitized so the embedded JSON is strictly valid,
  * the export is a single self-contained file that inlines code + reflection +
    window-style DATA (opens from file:// with no server, no CDN).

No FalkorDB / network / Flask needed.
Run: PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_lineage_viz.py
"""
import json
import os
import tempfile

from graphretr_opt.viz.lineage_viz import build_graph, export_static, render_html


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _write_run(d):
    run = os.path.join(d, "runs", "vizc")
    os.makedirs(run, exist_ok=True)
    seed = "seedsha0" + "0" * 56
    a = "accsha01" + "1" * 56
    b = "rejsha02" + "2" * 56

    def mv(recall, bad=False):
        return {"quality_recall_at_20": (float("inf") if bad else recall),
                "quality_hit_at_1": 0.1, "latency_s": 0.5, "code_complexity": 12.0}

    rows = [
        {"step": 0, "parent_sha": seed, "child_sha": a, "accepted": True,
         "admitted": True, "frontier_grew": True, "metric_vector": mv(0.34),
         "reason": "admitted to pool (new Pareto frontier)", "pool_size": 1,
         "change_summary": "k=20 -> k=40", "gate_tag": "fix", "edit_budget": 2,
         "tokens_step": 1000},
        {"step": 1, "parent_sha": a, "child_sha": b, "accepted": False,
         "admitted": False, "frontier_grew": False, "metric_vector": mv(0.0, bad=True),
         "reason": "not admitted (dominated)", "pool_size": 1,
         "change_summary": "k=40 -> k=200", "gate_tag": "fix", "edit_budget": 2,
         "tokens_step": 1200},
        {"step": 2, "parent_sha": a, "child_sha": None, "accepted": False,
         "admitted": False, "frontier_grew": False, "metric_vector": None,
         "reason": "mutator returned no usable candidate", "pool_size": 1,
         "change_summary": "(no candidate)", "gate_tag": "fix", "edit_budget": 2,
         "tokens_step": 0},
    ]
    with open(os.path.join(run, "lineage.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    open(os.path.join(run, "seed_used.py"), "w").write("def search(q, G):\n    return []\n")
    open(os.path.join(run, "step_000.py"), "w").write("def search(q, G):\n    return [1]  # accepted\n")
    open(os.path.join(run, "step_001.py"), "w").write("def search(q, G):\n    return [2]  # rejected\n")
    open(os.path.join(run, "reflection_000.md"), "w").write("# reflection 0\nprompt+response")
    return run, {"seed": seed, "a": a, "b": b}


def test_build_graph():
    with tempfile.TemporaryDirectory() as d:
        run, shas = _write_run(d)
        g = build_graph(run)
        by = {n["sha"]: n for n in g["nodes"]}
        _check("seed root present", shas["seed"] in by and by[shas["seed"]].get("is_seed"))
        _check("accepted node present + green-coded", by[shas["a"]]["accepted"] is True)
        _check("rejected node present + not accepted", by[shas["b"]]["accepted"] is False)
        _check("edges chain seed->a->b",
               {(e["src"], e["dst"]) for e in g["edges"]}
               == {(shas["seed"], shas["a"]), (shas["a"], shas["b"])})
        _check("no-candidate step folded onto its parent (a), not a node",
               len(by[shas["a"]]["no_cand_children"]) == 1)
        _check("code inlined for accepted node", "accepted" in (by[shas["a"]]["code"] or ""))
        _check("reflection inlined for step 0", "reflection 0" in (by[shas["a"]]["reflection"] or ""))
        _check("meta counts candidates (a,b) not seed/no-cand", g["meta"]["candidates"] == 2)
        _check("meta counts accepted", g["meta"]["accepted"] == 1)
        _check("meta counts no-candidate steps", g["meta"]["no_candidate_steps"] == 1)


def test_inf_is_sanitized_and_json_valid():
    with tempfile.TemporaryDirectory() as d:
        run, shas = _write_run(d)
        g = build_graph(run)
        by = {n["sha"]: n for n in g["nodes"]}
        _check("inf recall sanitized to None",
               by[shas["b"]]["metric_vector"]["quality_recall_at_20"] is None)
        # render must not raise (json.dumps allow_nan=False) and must embed the data
        html = render_html(g)
        _check("html embeds DATA payload", "const DATA = {" in html)
        _check("no raw Infinity/NaN tokens leaked into JSON",
               "Infinity" not in html and "NaN" not in html)


def test_export_static_self_contained():
    with tempfile.TemporaryDirectory() as d:
        run, _ = _write_run(d)
        out = export_static(run)
        _check("lineage.html written next to the trace",
               out == os.path.join(run, "lineage.html") and os.path.exists(out))
        text = open(out).read()
        _check("single self-contained doc", text.startswith("<!doctype html>"))
        _check("inlines candidate code (no fetch needed)", "# accepted" in text)
        _check("no external/CDN script or link tags",
               "src=\"http" not in text and "href=\"http" not in text
               and "<script src" not in text)


def main():
    test_build_graph()
    test_inf_is_sanitized_and_json_valid()
    test_export_static_self_contained()
    print("\nall lineage_viz tests passed")


if __name__ == "__main__":
    main()
