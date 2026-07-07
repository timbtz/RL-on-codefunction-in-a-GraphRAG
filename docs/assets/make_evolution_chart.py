#!/usr/bin/env python3
"""Evolution figure (light + dark SVG): (A) accepted-candidate recall@20 over
optimizer steps, run14_glm chained into run15 archipelago (from lineage.jsonl);
(B) ingestion+search co-optimization on the enterprise corpus (seed vs best).
"""
import os, sys

RUN14 = {2:0.5597,3:0.3750,4:0.6042,5:0.5597,7:0.4153,9:0.4194,10:0.6042,11:0.5597,
         14:0.5042,15:0.5014,16:0.2681,20:0.4486,23:0.4153,24:0.5597,26:0.5708,
         31:0.5708,32:0.6375,33:0.2181,34:0.4458,35:0.2792,36:0.5056,37:0.4986,39:0.5875}
RUN15 = {0:0.6097,1:0.5389,2:0.4681,3:0.5014,5:0.3806,8:0.4028,9:0.5708,10:0.4042,
         11:0.4431,12:0.4819,13:0.5708,14:0.5708,15:0.4819,16:0.4653,17:0.5042,
         18:0.5931,20:0.5375,21:0.5833,22:0.6153,25:0.6153,26:0.5708,28:0.5375}
SEED14, SEED15 = 0.4597, 0.5597
OFF = 43                      # run15 steps drawn at x = step + OFF
ENV14 = [(0,0.4597),(2,0.5597),(4,0.6042),(32,0.6375),(39,0.6375)]
ENV15 = [(0,0.6097),(22,0.6153),(28,0.6153)]

CORPUS = [("MCQ accuracy", 0.767, 0.967), ("retrieval-hit", 0.733, 1.000)]

THEMES = {
    "light": dict(c14="#2a78d6", c15="#1baf7a", base="#c9c8c0", ring="#ffffff",
                  ink="#24292f", ink2="#57606a", muted="#6e7781",
                  grid="#d8dee4", axis="#afb8c1", band="#f0f3f6"),
    "dark":  dict(c14="#3987e5", c15="#199e70", base="#484f58", ring="#0d1117",
                  ink="#e6edf3", ink2="#9198a1", muted="#8b949e",
                  grid="#30363d", axis="#484f58", band="#161b22"),
}
FONT = 'font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif"'

# ---- panel A geometry ----
A_X, A_Y, A_W, A_H = 52, 46, 600, 250
XMAX = 74                     # steps 0..39 run14, 43..71 run15
YMIN, YMAX = 0.15, 0.68

def ax(step): return A_X + A_W * step / XMAX
def ay(v):    return A_Y + A_H * (YMAX - v) / (YMAX - YMIN)

def panel_a(t):
    o = []
    # run15 region band
    o.append(f'<rect x="{ax(OFF-1.5):.1f}" y="{A_Y}" width="{ax(XMAX)-ax(OFF-1.5):.1f}" height="{A_H}" fill="{t["band"]}"/>')
    o.append(f'<text x="{A_X}" y="{A_Y-24}" {FONT} font-size="13" font-weight="600" fill="{t["ink"]}">A · Accepted candidates over optimizer steps — recall@20 on rotating gate sets</text>')
    o.append(f'<text x="{A_X}" y="{A_Y-9}" {FONT} font-size="11" fill="{t["muted"]}">run14 (GLM mutator, 40 steps)  →  champion seeds run15 archipelago island (32 steps)</text>')
    # gridlines
    for gv in (0.2, 0.3, 0.4, 0.5, 0.6):
        o.append(f'<line x1="{A_X}" y1="{ay(gv):.1f}" x2="{A_X+A_W}" y2="{ay(gv):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{A_X-7}" y="{ay(gv)+3.5:.1f}" {FONT} font-size="10.5" fill="{t["muted"]}" text-anchor="end">{gv:.1f}</text>')
    o.append(f'<line x1="{A_X}" y1="{A_Y}" x2="{A_X}" y2="{A_Y+A_H}" stroke="{t["axis"]}" stroke-width="1"/>')
    o.append(f'<line x1="{A_X}" y1="{A_Y+A_H}" x2="{A_X+A_W}" y2="{A_Y+A_H}" stroke="{t["axis"]}" stroke-width="1"/>')
    # x ticks: run14 0/10/20/30/40 ; run15 0/10/20/30 at offset
    for s in (0, 10, 20, 30, 40):
        o.append(f'<text x="{ax(s):.1f}" y="{A_Y+A_H+15}" {FONT} font-size="10.5" fill="{t["muted"]}" text-anchor="middle">{s}</text>')
    for s in (0, 10, 20, 28):
        o.append(f'<text x="{ax(s+OFF):.1f}" y="{A_Y+A_H+15}" {FONT} font-size="10.5" fill="{t["muted"]}" text-anchor="middle">{s}</text>')
    o.append(f'<text x="{ax(20):.1f}" y="{A_Y+A_H+30}" {FONT} font-size="10.5" fill="{t["ink2"]}" text-anchor="middle">run14 step</text>')
    o.append(f'<text x="{ax(OFF+14):.1f}" y="{A_Y+A_H+30}" {FONT} font-size="10.5" fill="{t["ink2"]}" text-anchor="middle">run15 step</text>')
    # seed-chain connector
    o.append(f'<line x1="{ax(32):.1f}" y1="{ay(0.6375):.1f}" x2="{ax(OFF):.1f}" y2="{ay(0.6097):.1f}" stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="3 3"/>')
    # running-best envelopes (step lines)
    for env, col in ((ENV14, t["c14"]), (ENV15, t["c15"])):
        pts, prev = [], None
        for s, v in env:
            x = ax(s if col == t["c14"] else s + OFF)
            if prev is not None:
                pts.append(f"L {x:.1f} {prev:.1f}")
            pts.append(f'{"M" if prev is None else "L"} {x:.1f} {ay(v):.1f}')
            prev = ay(v)
        o.append(f'<path d="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" opacity="0.85"/>')
    # seed markers
    for s, v, col, lbl, anchor in ((0, SEED14, t["c14"], "seed 0.46", "start"),
                                   (OFF, SEED15, t["c15"], "seeded at 0.56", "start")):
        x = ax(s)
        o.append(f'<rect x="{x-4:.1f}" y="{ay(v)-4:.1f}" width="8" height="8" transform="rotate(45 {x:.1f} {ay(v):.1f})" fill="{col}" stroke="{t["ring"]}" stroke-width="2"/>')
    o.append(f'<text x="{ax(0)+7:.1f}" y="{ay(SEED14)+14:.1f}" {FONT} font-size="10.5" fill="{t["ink2"]}">seed 0.46</text>')
    o.append(f'<text x="{ax(OFF)+7:.1f}" y="{ay(SEED15)+16:.1f}" {FONT} font-size="10.5" fill="{t["ink2"]}">seeded at 0.56</text>')
    # points
    for d, col, off in ((RUN14, t["c14"], 0), (RUN15, t["c15"], OFF)):
        for s, v in sorted(d.items()):
            o.append(f'<circle cx="{ax(s+off):.1f}" cy="{ay(v):.1f}" r="4" fill="{col}" stroke="{t["ring"]}" stroke-width="2"/>')
    # peak annotations
    o.append(f'<text x="{ax(32):.1f}" y="{ay(0.6375)-9:.1f}" {FONT} font-size="10.5" font-weight="600" fill="{t["ink"]}" text-anchor="middle">0.638</text>')
    o.append(f'<text x="{ax(OFF+22):.1f}" y="{ay(0.6153)-9:.1f}" {FONT} font-size="10.5" font-weight="600" fill="{t["ink"]}" text-anchor="middle">0.615</text>')
    # legend
    lx = A_X + 8
    ly = A_Y + A_H - 16
    o.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="{t["c14"]}"/><text x="{lx+9}" y="{ly+3.5}" {FONT} font-size="11" fill="{t["ink2"]}">run14 accepted</text>')
    o.append(f'<circle cx="{lx+118}" cy="{ly}" r="4" fill="{t["c15"]}"/><text x="{lx+127}" y="{ly+3.5}" {FONT} font-size="11" fill="{t["ink2"]}">run15 accepted (island b0_r0)</text>')
    return "\n".join(o)

# ---- panel B geometry ----
B_X, B_Y, B_W, B_H = 726, 46, 220, 250

def by(v): return B_Y + B_H * (1 - v)

def panel_b(t):
    o = []
    o.append(f'<text x="{B_X}" y="{B_Y-24}" {FONT} font-size="13" font-weight="600" fill="{t["ink"]}">B · Ingestion + search co-optimized</text>')
    o.append(f'<text x="{B_X}" y="{B_Y-9}" {FONT} font-size="11" fill="{t["muted"]}">5-source enterprise corpus, 81-MCQ exam</text>')
    for gv in (0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{B_X}" y1="{by(gv):.1f}" x2="{B_X+B_W}" y2="{by(gv):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{B_X-7}" y="{by(gv)+3.5:.1f}" {FONT} font-size="10.5" fill="{t["muted"]}" text-anchor="end">{gv:g}</text>')
    o.append(f'<line x1="{B_X}" y1="{B_Y+B_H}" x2="{B_X+B_W}" y2="{B_Y+B_H}" stroke="{t["axis"]}" stroke-width="1"/>')
    group_w = B_W / 2
    bw, gap = 34, 8
    for gi, (label, seed, best) in enumerate(CORPUS):
        cx = B_X + group_w * gi + group_w / 2
        x1 = cx - bw - gap / 2
        x2 = cx + gap / 2
        for x, v, col, wt in ((x1, seed, t["base"], ""), (x2, best, t["c14"], ' font-weight="600"')):
            h = B_H * v
            r = 4
            o.append(f'<path d="M {x:.1f} {by(v)+r:.1f} a {r} {r} 0 0 1 {r} -{r} h {bw-2*r} a {r} {r} 0 0 1 {r} {r} v {h-r:.1f} h -{bw} z" fill="{col}"/>')
            if v >= 0.95:  # label inside the bar cap to avoid the subtitle
                o.append(f'<text x="{x+bw/2:.1f}" y="{by(v)+14:.1f}" {FONT} font-size="11"{wt} fill="{t["ring"]}" text-anchor="middle">{v:.2f}</text>')
            else:
                o.append(f'<text x="{x+bw/2:.1f}" y="{by(v)-5:.1f}" {FONT} font-size="11"{wt} fill="{t["ink"] if wt else t["ink2"]}" text-anchor="middle">{v:.2f}</text>')
        o.append(f'<text x="{cx:.1f}" y="{B_Y+B_H+15}" {FONT} font-size="11" fill="{t["ink2"]}" text-anchor="middle">{label}</text>')
    ly = B_Y + B_H + 32
    o.append(f'<rect x="{B_X}" y="{ly-8}" width="10" height="10" fill="{t["base"]}" rx="2"/><text x="{B_X+15}" y="{ly+1}" {FONT} font-size="11" fill="{t["ink2"]}">hand-written seed</text>')
    o.append(f'<rect x="{B_X+124}" y="{ly-8}" width="10" height="10" fill="{t["c14"]}" rx="2"/><text x="{B_X+139}" y="{ly+1}" {FONT} font-size="11" fill="{t["ink2"]}">co-optimized</text>')
    return "\n".join(o)

W, H = 990, 356
outdir = sys.argv[1] if len(sys.argv) > 1 else "."
for mode, t in THEMES.items():
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Optimizer evolution: accepted candidate recall over steps across two chained campaigns, and seed versus co-optimized scores on the enterprise corpus">']
    svg.append(panel_a(t))
    svg.append(panel_b(t))
    svg.append('</svg>')
    p = os.path.join(outdir, f"cooptimize-evolution-{mode}.svg")
    with open(p, "w") as f:
        f.write("\n".join(svg))
    print("wrote", p)
