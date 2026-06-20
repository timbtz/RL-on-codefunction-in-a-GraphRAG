"""lineage_viz -- turn a campaign's lineage.jsonl into a self-contained HTML
phylogeny + metric view (Phase 3).

graphretr already writes an RL-shaped per-step trace (`lineage.jsonl`: state =
parent_sha, action = the edit, reward = the metric vector, accepted AND rejected
steps) plus the full candidate code (`step_NNN.py`) and prompt/response
(`reflection_NNN.md`) as side files. The only gap the plan identifies is the
frontend, so this is a pure read-side consumer:

  * build a parent_sha -> child_sha graph from the JSONL (no islands -> color by
    accepted / rejected, position by step x recall@20),
  * INLINE the matching code + reflection per node, and `window.STATIC_DATA`, so
    the exported HTML is a single serverless, auditable artifact (openEvolve's
    static-export pattern) -- it opens from file:// with no network and no CDN,
  * sanitize inf/NaN so the embedded JSON stays valid.

A Flask live-serve mode is optional (`serve()`); the static export is the default
and needs no dependencies beyond the stdlib.
"""
import html
import json
import math
import os

PRIMARY_KEY = "quality_recall_at_20"  # MetricVector.as_flat() name for recall@20


def _sanitize(obj):
    """Recursively replace non-finite floats (inf/NaN) with None so json.dumps
    produces strictly-valid JSON (openEvolve's sanitize_program_for_visualization
    equivalent). A misbehaving candidate must not corrupt the whole artifact."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def load_lineage(run_dir):
    """Read runs/<campaign>/lineage.jsonl -> list[row] (skips blank/torn lines)."""
    path = os.path.join(run_dir, "lineage.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no lineage.jsonl in {run_dir} -- run `optimize` first")
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # tolerate a torn final line (e.g. a crash mid-append); skip it.
            continue
    return rows


def _read_side(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    try:
        return open(path).read()
    except OSError:
        return None


def build_graph(run_dir, inline_sources=True):
    """Build {nodes, edges, meta} from lineage.jsonl.

    A node is one PROPOSED candidate (a row with a child_sha; rejected candidates
    keep their sha too). Rows with no candidate (parse/agent failures) become a
    counter on their parent rather than a dangling node. The seed is the root."""
    rows = load_lineage(run_dir)
    nodes, edges = {}, []
    no_cand = []
    root_sha = rows[0]["parent_sha"] if rows else None

    for r in rows:
        csha = r.get("child_sha")
        mv = r.get("metric_vector") or {}
        recall = mv.get(PRIMARY_KEY)
        if csha is None:
            no_cand.append(r)
            continue
        node = {
            "sha": csha,
            "short": csha[:8],
            "step": r.get("step"),
            "parent_sha": r.get("parent_sha"),
            "accepted": bool(r.get("accepted")),
            "admitted": bool(r.get("admitted")),
            "frontier_grew": bool(r.get("frontier_grew")),
            "recall": recall,
            "metric_vector": mv,
            "reason": r.get("reason"),
            "change_summary": r.get("change_summary"),
            "gate_tag": r.get("gate_tag"),
            "edit_budget": r.get("edit_budget"),
            "tokens_step": r.get("tokens_step"),
            "pool_size": r.get("pool_size"),
        }
        if inline_sources:
            step = r.get("step")
            node["code"] = _read_side(run_dir, f"step_{step:03d}.py")
            node["reflection"] = _read_side(run_dir, f"reflection_{step:03d}.md")
        # de-dup on sha (a re-proposed identical candidate / cache hit): keep the
        # first occurrence but remember it recurred.
        if csha in nodes:
            nodes[csha]["recurred"] = nodes[csha].get("recurred", 1) + 1
        else:
            nodes[csha] = node
        if r.get("parent_sha"):
            edges.append({"src": r["parent_sha"], "dst": csha,
                          "accepted": bool(r.get("accepted"))})

    # attach no-candidate failures to their parent (audit, not graph clutter)
    fails_by_parent = {}
    for r in no_cand:
        fails_by_parent.setdefault(r.get("parent_sha"), []).append(
            {"step": r.get("step"), "reason": r.get("reason")})

    # the seed root carries no metric in the trace; place it at step -1.
    root = {
        "sha": root_sha, "short": (root_sha or "seed")[:8], "step": -1,
        "parent_sha": None, "accepted": True, "admitted": True,
        "frontier_grew": False, "recall": None, "metric_vector": {},
        "reason": "seed", "change_summary": "(seed program)", "is_seed": True,
        "code": _read_side(run_dir, "seed_used.py") if inline_sources else None,
    }
    if root_sha and root_sha not in nodes:
        nodes[root_sha] = root
    elif root_sha:
        nodes[root_sha].setdefault("is_seed", True)

    for sha, node in nodes.items():
        node["no_cand_children"] = fails_by_parent.get(sha, [])

    accepted = sum(1 for n in nodes.values() if n.get("accepted") and not n.get("is_seed"))
    return {
        "nodes": _sanitize(list(nodes.values())),
        "edges": edges,
        "meta": {
            "campaign": os.path.basename(os.path.normpath(run_dir)),
            "rows": len(rows),
            "candidates": len([n for n in nodes.values() if not n.get("is_seed")]),
            "accepted": accepted,
            "no_candidate_steps": len(no_cand),
            "primary_key": PRIMARY_KEY,
        },
    }


# --------------------------------------------------------------------- rendering

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>graphretr lineage — __CAMPAIGN__</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 13px/1.45 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         display: grid; grid-template-columns: 1fr 460px; height: 100vh; }
  header { grid-column: 1 / -1; padding: 8px 14px; background: #1b2330; color: #e8eef7;
           display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }
  header b { color: #fff; } header .k { color: #8fb6e6; }
  #plot { position: relative; overflow: auto; border-right: 1px solid #ccc; }
  svg { display: block; background: #fafbfc; }
  circle.node { cursor: pointer; stroke: #333; stroke-width: 1; }
  circle.node.sel { stroke: #ff8c00; stroke-width: 3; }
  .acc { fill: #2e9e5b; } .rej { fill: #d9d9d9; } .seed { fill: #3b6fb0; }
  .front { stroke: #b8860b; stroke-width: 2.5; }
  line.edge { stroke: #b9c2cf; stroke-width: 1; }
  line.edge.acc { stroke: #2e9e5b; stroke-width: 1.6; }
  text.axis { fill: #888; font-size: 11px; }
  #side { overflow: auto; padding: 12px 14px; }
  #side h2 { margin: 0 0 6px; font-size: 15px; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
           color: #fff; margin-right: 5px; }
  .badge.a { background: #2e9e5b; } .badge.r { background: #b0451f; }
  .badge.f { background: #b8860b; }
  table.mv { border-collapse: collapse; margin: 8px 0; }
  table.mv td { padding: 1px 10px 1px 0; } table.mv td.n { text-align: right; font-variant-numeric: tabular-nums; }
  pre { background: #f4f5f7; padding: 8px; overflow: auto; max-height: 320px; border-radius: 4px;
        white-space: pre-wrap; word-break: break-word; }
  details > summary { cursor: pointer; margin: 6px 0; font-weight: 600; }
  #search { width: 100%; padding: 5px; box-sizing: border-box; margin-bottom: 8px; }
  .muted { color: #888; }
</style></head>
<body>
<header>
  <b>graphretr lineage</b> <span>__CAMPAIGN__</span>
  <span><span class="k">candidates</span> __N_CAND__</span>
  <span><span class="k">accepted</span> __N_ACC__</span>
  <span><span class="k">steps</span> __N_ROWS__</span>
  <span><span class="k">no-candidate</span> __N_NOCAND__</span>
  <span class="muted">x = step · y = recall@20 · green = accepted · gray = rejected · gold ring = new frontier</span>
</header>
<div id="plot"><svg id="svg"></svg></div>
<div id="side">
  <input id="search" placeholder="filter by sha / step / reason…">
  <div id="detail"><p class="muted">Click a node to inspect its code, reflection, and metrics.</p></div>
  <div id="list"></div>
</div>
<script>
const DATA = __DATA__;
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const nodes = DATA.nodes, edges = DATA.edges;
const recalls = nodes.map(n => n.recall).filter(v => v!=null);
const rMin = recalls.length ? Math.min(...recalls) : 0, rMax = recalls.length ? Math.max(...recalls) : 1;
const steps = nodes.map(n => n.step);
const sMin = Math.min(...steps), sMax = Math.max(...steps);
const W = Math.max(720, (sMax - sMin + 2) * 46), H = 560, PAD = 48;
const x = s => PAD + (s - sMin) * (W - 2*PAD) / Math.max(1, sMax - sMin);
const y = r => (r==null) ? (H - PAD/2)
              : H - PAD - (r - rMin) * (H - 2*PAD) / Math.max(1e-9, rMax - rMin);
const bySha = Object.fromEntries(nodes.map(n => [n.sha, n]));

const svg = document.getElementById("svg");
svg.setAttribute("width", W); svg.setAttribute("height", H);
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs) { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }

// axis ticks for recall
for (let i = 0; i <= 4; i++) {
  const r = rMin + (rMax - rMin) * i / 4;
  const yy = y(r);
  svg.appendChild(el("line", {x1: PAD-6, y1: yy, x2: W-PAD, y2: yy, stroke: "#eee"}));
  const t = el("text", {x: 4, y: yy+3, class: "axis"}); t.textContent = r.toFixed(3);
  svg.appendChild(t);
}
// edges
for (const e of edges) {
  const a = bySha[e.src], b = bySha[e.dst];
  if (!a || !b) continue;
  svg.appendChild(el("line", {x1: x(a.step), y1: y(a.recall), x2: x(b.step), y2: y(b.recall),
    class: "edge" + (e.accepted ? " acc" : "")}));
}
// nodes
const circles = {};
for (const n of nodes) {
  const cls = n.is_seed ? "seed" : (n.accepted ? "acc" : "rej");
  const c = el("circle", {cx: x(n.step), cy: y(n.recall),
    r: n.is_seed ? 7 : 6, class: "node " + cls + (n.frontier_grew ? " front" : "")});
  c.addEventListener("click", () => select(n.sha));
  c.appendChild(el("title", {})).textContent =
    `step ${n.step} · ${n.short} · recall ${n.recall==null?"—":n.recall.toFixed(4)} · ${n.reason||""}`;
  svg.appendChild(c); circles[n.sha] = c;
}

function mvTable(mv) {
  const keys = Object.keys(mv||{});
  if (!keys.length) return "<p class='muted'>no metric vector</p>";
  return "<table class='mv'>" + keys.map(k =>
    `<tr><td>${esc(k)}</td><td class='n'>${mv[k]==null?"—":(+mv[k]).toFixed(4)}</td></tr>`).join("") + "</table>";
}
function select(sha) {
  for (const s in circles) circles[s].classList.toggle("sel", s===sha);
  const n = bySha[sha]; if (!n) return;
  const badges = (n.is_seed ? "<span class='badge'>seed</span>" : "")
    + (n.accepted ? "<span class='badge a'>accepted</span>" : "<span class='badge r'>rejected</span>")
    + (n.admitted ? "<span class='badge a'>admitted</span>" : "")
    + (n.frontier_grew ? "<span class='badge f'>new frontier</span>" : "");
  const fails = (n.no_cand_children||[]).length
    ? `<p class='muted'>${n.no_cand_children.length} no-candidate step(s) followed from here</p>` : "";
  document.getElementById("detail").innerHTML =
    `<h2>step ${n.step} · ${esc(n.short)}</h2>${badges}
     <p><b>reason:</b> ${esc(n.reason)}</p>
     <p><b>change:</b> ${esc(n.change_summary)}</p>
     <p class='muted'>gate ${esc(n.gate_tag)} · edit budget ${esc(n.edit_budget)} ·
        pool ${esc(n.pool_size)} · tokens ${esc(n.tokens_step)}</p>
     ${fails}
     ${mvTable(n.metric_vector)}
     ${n.code!=null ? `<details open><summary>candidate code</summary><pre>${esc(n.code)}</pre></details>` : ""}
     ${n.reflection!=null ? `<details><summary>reflection (prompt + response)</summary><pre>${esc(n.reflection)}</pre></details>` : ""}`;
  document.getElementById("side").scrollTop = 0;
}

function renderList(filter) {
  filter = (filter||"").toLowerCase();
  const rows = nodes.filter(n => !n.is_seed).sort((a,b)=>a.step-b.step).filter(n =>
    !filter || (n.short+" "+n.step+" "+(n.reason||"")).toLowerCase().includes(filter));
  document.getElementById("list").innerHTML =
    "<h2>steps</h2>" + rows.map(n =>
      `<div style="padding:2px 0;cursor:pointer" onclick="window._sel('${n.sha}')">
        <span class="badge ${n.accepted?'a':'r'}">${n.step}</span>
        ${esc(n.short)} · recall ${n.recall==null?"—":n.recall.toFixed(4)}
        <span class="muted">${esc((n.reason||"").slice(0,48))}</span></div>`).join("");
}
window._sel = select;
document.getElementById("search").addEventListener("input", e => renderList(e.target.value));
renderList("");
</script>
</body></html>
"""


def render_html(graph):
    meta = graph["meta"]
    data = json.dumps(graph, allow_nan=False)
    out = _TEMPLATE
    out = out.replace("__DATA__", data)
    out = out.replace("__CAMPAIGN__", html.escape(meta["campaign"]))
    out = out.replace("__N_CAND__", str(meta["candidates"]))
    out = out.replace("__N_ACC__", str(meta["accepted"]))
    out = out.replace("__N_ROWS__", str(meta["rows"]))
    out = out.replace("__N_NOCAND__", str(meta["no_candidate_steps"]))
    return out


def export_static(run_dir, out_path=None):
    """Write a single self-contained runs/<campaign>/lineage.html. -> path."""
    graph = build_graph(run_dir, inline_sources=True)
    out_path = out_path or os.path.join(run_dir, "lineage.html")
    with open(out_path, "w") as f:
        f.write(render_html(graph))
    return out_path


def serve(run_dir, host="127.0.0.1", port=8000):
    """Optional live server: re-reads lineage.jsonl on every load so a running
    campaign is watchable. Flask is only imported here (static export needs none)."""
    try:
        from flask import Flask, Response
    except ImportError as e:
        raise SystemExit("flask is not installed; use the static export instead "
                         "(omit --serve)") from e
    app = Flask(__name__)

    @app.route("/")
    def index():
        return Response(render_html(build_graph(run_dir, inline_sources=True)),
                        mimetype="text/html")

    print(f"[viz] serving {run_dir} at http://{host}:{port}  (Ctrl-C to stop)")
    app.run(host=host, port=port)
