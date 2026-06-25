// ============================================================
// renderHtml.ts — GraphModel → self-contained interactive HTML page
// ============================================================
//
// Emits a single HTML document (Cytoscape.js loaded from a CDN) that draws
// each module as a compound box, the nodes inside it, intra-module edges, and
// cross-module edges between boxes. Properties for nodes and relationships can
// be toggled live in the page; `showProperties` only sets the initial state.
// Buttons export the current view as PNG or SVG.

import type { GraphModel } from "./buildModel.js";

export interface RenderOptions {
  /** Initial state of the property display. Default: true. */
  showProperties?: boolean;
  /** Page <title> and on-page heading. Default: "Graph Module Map". */
  title?: string;
}

/** Embed JSON safely inside a <script> tag (neutralize `</script>`). */
function embed(data: unknown): string {
  return JSON.stringify(data).replace(/</g, "\\u003c");
}

export function renderHtml(model: GraphModel, options: RenderOptions = {}): string {
  const showProperties = options.showProperties ?? true;
  const title = options.title ?? "Graph Module Map";

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${escapeHtml(title)}</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
<script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<script src="https://unpkg.com/cytoscape-svg@0.4.0/cytoscape-svg.js"></script>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, sans-serif; }
  #toolbar {
    position: fixed; top: 0; left: 0; right: 0; height: 52px; z-index: 10;
    display: flex; align-items: center; gap: 8px; padding: 0 14px;
    background: #1e2230; color: #e8eaf0; border-bottom: 1px solid #333a4d;
  }
  #toolbar h1 { font-size: 15px; font-weight: 600; margin: 0 12px 0 0; white-space: nowrap; }
  #toolbar .spacer { flex: 1; }
  #toolbar button, #toolbar label.toggle {
    font: inherit; font-size: 13px; cursor: pointer; border: 1px solid #404862; border-radius: 6px;
    background: #2b3142; color: #e8eaf0; padding: 6px 11px;
  }
  #toolbar button:hover { background: #353c52; }
  #toolbar label.toggle { display: inline-flex; align-items: center; gap: 7px; user-select: none; }
  #toolbar input[type=search] {
    font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px;
    border: 1px solid #404862; background: #2b3142; color: #e8eaf0; width: 180px;
  }
  #cy { position: fixed; top: 52px; left: 0; right: 0; bottom: 0; background: #0f1117; }
  #modules {
    position: fixed; top: 64px; left: 14px; z-index: 10; width: 220px;
    max-height: calc(100% - 78px); display: flex; flex-direction: column;
    background: rgba(30,34,48,.94); color: #e8eaf0; border: 1px solid #333a4d;
    border-radius: 8px; font-size: 13px;
  }
  #modules .head {
    display: flex; align-items: center; gap: 6px; padding: 9px 11px;
    border-bottom: 1px solid #333a4d;
  }
  #modules .head b { flex: 1; font-size: 12px; }
  #modules .head button {
    font: inherit; font-size: 11px; cursor: pointer; border: 1px solid #404862;
    border-radius: 5px; background: #2b3142; color: #e8eaf0; padding: 3px 7px;
  }
  #modules .head button:hover { background: #353c52; }
  #modules .list { overflow-y: auto; padding: 6px 0; }
  #modules label {
    display: flex; align-items: center; gap: 8px; padding: 4px 11px;
    cursor: pointer; user-select: none; line-height: 1.3;
  }
  #modules label:hover { background: rgba(255,255,255,.05); }
  #modules label.off { opacity: 0.5; }
  #modules .swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  #modules .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #legend {
    position: fixed; right: 14px; bottom: 14px; z-index: 10; max-width: 260px;
    background: rgba(30,34,48,.94); color: #e8eaf0; border: 1px solid #333a4d;
    border-radius: 8px; padding: 10px 12px; font-size: 12px; line-height: 1.7;
  }
  #legend b { display: block; margin-bottom: 4px; font-size: 12px; }
  #legend .row { display: flex; align-items: center; gap: 8px; }
  #legend .swatch { width: 22px; height: 0; border-top-width: 3px; border-top-style: solid; }
  #legend .box { width: 14px; height: 14px; border-radius: 3px; border: 1.5px solid; }
</style>
</head>
<body>
<div id="toolbar">
  <h1>${escapeHtml(title)}</h1>
  <label class="toggle"><input type="checkbox" id="toggle-props" ${showProperties ? "checked" : ""}/> Properties</label>
  <input type="search" id="search" placeholder="Highlight node…" />
  <button id="fit">Fit</button>
  <button id="relayout">Re-layout</button>
  <span class="spacer"></span>
  <button id="png">PNG</button>
  <button id="svg">SVG</button>
</div>
<div id="cy"></div>
<div id="modules">
  <div class="head">
    <b>Modules</b>
    <button id="mod-all-on">All on</button>
    <button id="mod-all-off">All off</button>
  </div>
  <div class="list" id="modules-list"></div>
</div>
<div id="legend">
  <b>Legend</b>
  <div class="row"><span class="box" style="background:#2a3550;border-color:#5b78c7"></span> owned node</div>
  <div class="row"><span class="box" style="background:#2f2a3f;border-color:#9b6bd6"></span> shared node</div>
  <div class="row"><span class="box" style="background:transparent;border-color:#e0b341;border-style:dashed"></span> root / included</div>
  <div class="row"><span class="swatch" style="border-top-color:#8aa0d8"></span> intra-module</div>
  <div class="row"><span class="swatch" style="border-top-color:#4fae7a"></span> cross-module (out)</div>
  <div class="row"><span class="swatch" style="border-top-color:#d77a7a;border-top-style:dashed"></span> cross-module (in)</div>
  <div class="row"><span class="swatch" style="border-top-color:#6b7280;border-top-style:dotted"></span> includes</div>
</div>

<script>
const MODEL = ${embed(model)};
let SHOW_PROPS = ${showProperties ? "true" : "false"};

cytoscape.use(window.cytoscapeFcose);
if (window.cytoscapeSvg) cytoscape.use(window.cytoscapeSvg);

// ---- Per-module palette ----
const palette = [
  "#5b78c7","#4fae7a","#c77b5b","#9b6bd6","#3fa0b8","#c75b9a","#b8a23f","#5bc7b4",
];
const moduleColor = {};
MODEL.modules.forEach((m, i) => { moduleColor[m.name] = palette[i % palette.length]; });

// ---- Label builders ----
function nodeShort(n) {
  const head = n.isRoot ? "★ " + n.key : n.key;
  const labelLine = (n.labels.length && (n.labels.length > 1 || n.labels[0] !== n.key))
    ? "\\n:" + n.labels.join(" :") : "";
  return head + labelLine;
}
function nodeFull(n) {
  let s = nodeShort(n);
  if (n.includedFrom) s += "\\n(from " + n.includedFrom + ")";
  if (n.properties.length) {
    s += "\\n────────";
    for (const p of n.properties) s += "\\n" + p.name + ": " + p.type;
  }
  return s;
}
function edgeShort(e) {
  return e.kind === "include" ? "includes" : e.name + "  [" + e.cardinality + "]";
}
function edgeFull(e) {
  let s = edgeShort(e);
  if (e.relProperties.length) {
    s += "\\n{ " + e.relProperties.map(p => p.name + ": " + p.type).join(", ") + " }";
  }
  return s;
}

// ---- Build Cytoscape elements ----
const elements = [];
const seenParents = new Set();
for (const n of MODEL.nodes) {
  if (!seenParents.has(n.moduleName)) {
    seenParents.add(n.moduleName);
    elements.push({ data: { id: "grp::" + n.moduleName, label: n.moduleName, isParent: true,
      module: n.moduleName, color: moduleColor[n.moduleName] || "#888" } });
  }
  elements.push({ data: {
    id: n.id, parent: "grp::" + n.moduleName, module: n.moduleName,
    short: nodeShort(n), full: nodeFull(n), label: SHOW_PROPS ? nodeFull(n) : nodeShort(n),
    color: moduleColor[n.moduleName] || "#888",
    isRoot: n.isRoot ? 1 : 0, included: n.includedFrom ? 1 : 0,
  }});
}
for (const e of MODEL.edges) {
  elements.push({ data: {
    id: e.id, source: e.source, target: e.target, ekind: e.kind,
    short: edgeShort(e), full: edgeFull(e), label: SHOW_PROPS ? edgeFull(e) : edgeShort(e),
  }});
}

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements,
  wheelSensitivity: 0.2,
  style: [
    { selector: "node", style: {
      "background-color": "#2a3550", "border-color": "data(color)", "border-width": 1.5,
      "shape": "round-rectangle", "label": "data(label)", "color": "#e8eaf0",
      "font-size": 11, "font-family": "ui-monospace, Menlo, Consolas, monospace",
      "text-wrap": "wrap", "text-valign": "center", "text-halign": "center",
      "padding": "8px", "width": "label", "height": "label", "text-max-width": 240,
    }},
    { selector: "node[?isRoot]", style: { "border-width": 3, "border-color": "#e0b341" } },
    { selector: "node[?included]", style: { "border-style": "dashed", "background-opacity": 0.55 } },
    { selector: "node:parent", style: {
      "background-color": "data(color)", "background-opacity": 0.08,
      "border-color": "data(color)", "border-width": 2, "shape": "round-rectangle",
      "label": "data(label)", "text-valign": "top", "text-halign": "center",
      "font-size": 15, "font-weight": "bold", "color": "data(color)",
      "padding": "22px", "text-margin-y": -4,
    }},
    { selector: "edge", style: {
      "width": 1.6, "line-color": "#8aa0d8", "target-arrow-color": "#8aa0d8",
      "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": 0.9,
      "label": "data(label)", "color": "#aeb6c9", "font-size": 9.5,
      "font-family": "ui-monospace, Menlo, Consolas, monospace",
      "text-wrap": "wrap", "text-background-color": "#0f1117",
      "text-background-opacity": 0.85, "text-background-padding": 2,
    }},
    { selector: 'edge[ekind = "outbound"]', style: {
      "line-color": "#4fae7a", "target-arrow-color": "#4fae7a", "width": 2.2 }},
    { selector: 'edge[ekind = "inbound"]', style: {
      "line-color": "#d77a7a", "target-arrow-color": "#d77a7a", "line-style": "dashed", "width": 2 }},
    { selector: 'edge[ekind = "include"]', style: {
      "line-color": "#6b7280", "target-arrow-color": "#6b7280", "line-style": "dotted",
      "target-arrow-shape": "diamond", "width": 1.4 }},
    { selector: ".dim", style: { "opacity": 0.12 } },
    { selector: ".hit", style: { "border-color": "#ffd54a", "border-width": 4 } },
    { selector: ".mod-off", style: { "display": "none" } },
  ],
  layout: layoutOpts(),
});

function layoutOpts() {
  return {
    name: "fcose", quality: "proof", animate: false, randomize: true,
    nodeSeparation: 90, idealEdgeLength: 130, nodeRepulsion: 9000,
    gravity: 0.25, padding: 40, packComponents: true,
  };
}

function applyLabels() {
  cy.batch(() => {
    cy.nodes().forEach(n => { if (!n.isParent()) n.data("label", SHOW_PROPS ? n.data("full") : n.data("short")); });
    cy.edges().forEach(e => e.data("label", SHOW_PROPS ? e.data("full") : e.data("short")));
  });
}

// ---- Controls ----
document.getElementById("toggle-props").addEventListener("change", (ev) => {
  SHOW_PROPS = ev.target.checked; applyLabels();
});
document.getElementById("fit").addEventListener("click", () => cy.fit(undefined, 40));
document.getElementById("relayout").addEventListener("click", () => cy.layout(layoutOpts()).run());
document.getElementById("search").addEventListener("input", (ev) => {
  const q = ev.target.value.trim().toLowerCase();
  cy.elements().removeClass("dim hit");
  if (!q) return;
  const hits = cy.nodes().filter(n => !n.isParent() &&
    ((n.data("full") || "").toLowerCase().includes(q)));
  if (!hits.length) return;
  const keep = hits.union(hits.connectedEdges()).union(hits.connectedEdges().connectedNodes());
  cy.elements().difference(keep).addClass("dim");
  hits.addClass("hit");
});

// ---- Module visibility ----
const hiddenModules = new Set();

function applyModuleVisibility() {
  cy.batch(() => {
    cy.nodes().forEach(n => n.toggleClass("mod-off", hiddenModules.has(n.data("module"))));
    // An edge is hidden if either endpoint's module is hidden (avoids dangling
    // cross-module edges when only one side is turned off).
    cy.edges().forEach(e =>
      e.toggleClass("mod-off", e.source().hasClass("mod-off") || e.target().hasClass("mod-off")));
  });
}

const moduleList = document.getElementById("modules-list");
const moduleCheckboxes = {};
MODEL.modules.forEach(m => {
  const label = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox"; cb.checked = true; cb.dataset.module = m.name;
  const swatch = document.createElement("span");
  swatch.className = "swatch"; swatch.style.background = moduleColor[m.name] || "#888";
  const name = document.createElement("span");
  name.className = "name"; name.textContent = m.name; name.title = m.name;
  label.append(cb, swatch, name);
  cb.addEventListener("change", () => {
    if (cb.checked) hiddenModules.delete(m.name); else hiddenModules.add(m.name);
    label.classList.toggle("off", !cb.checked);
    applyModuleVisibility();
  });
  moduleList.appendChild(label);
  moduleCheckboxes[m.name] = { cb, label };
});

function setAllModules(on) {
  hiddenModules.clear();
  for (const m of MODEL.modules) {
    const { cb, label } = moduleCheckboxes[m.name];
    cb.checked = on; label.classList.toggle("off", !on);
    if (!on) hiddenModules.add(m.name);
  }
  applyModuleVisibility();
}
document.getElementById("mod-all-on").addEventListener("click", () => setAllModules(true));
document.getElementById("mod-all-off").addEventListener("click", () => setAllModules(false));

function download(name, dataUrlOrBlob) {
  const a = document.createElement("a");
  a.download = name;
  a.href = (typeof dataUrlOrBlob === "string") ? dataUrlOrBlob : URL.createObjectURL(dataUrlOrBlob);
  a.click();
  if (typeof dataUrlOrBlob !== "string") URL.revokeObjectURL(a.href);
}
document.getElementById("png").addEventListener("click", () =>
  download("graph-modules.png", cy.png({ full: true, scale: 2, bg: "#0f1117" })));
document.getElementById("svg").addEventListener("click", () => {
  if (!cy.svg) { alert("SVG export unavailable (cytoscape-svg failed to load)."); return; }
  const svg = cy.svg({ full: true, bg: "#0f1117" });
  download("graph-modules.svg", new Blob([svg], { type: "image/svg+xml" }));
});

cy.ready(() => cy.fit(undefined, 40));
</script>
</body>
</html>
`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
