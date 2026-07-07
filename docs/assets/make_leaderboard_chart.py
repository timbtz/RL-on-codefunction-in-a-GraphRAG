#!/usr/bin/env python3
"""STARK-Prime leaderboard comparison chart (light + dark SVG) for the README.

Baselines: official STaRK leaderboard (synthesized, full test set), from
huggingface.co/spaces/snap-stanford/stark-leaderboard app.py.
Ours: 900-query locked test split (run14 balanced / run15 recall-max champions).
"""
import sys, os

# (label, recall@20, hit@1, ours?)
ROWS = [
    ("EvoRetrieve (evolved search)", 52.9, 28.2, "ours1"),
    ("AvaTaR (GPT-4-turbo)",      42.23, 20.10, None),
    ("AvaTaR (Claude-3-Opus)",    39.31, 18.44, None),
    ("GritLM-7b",                 39.09, 15.57, None),
    ("multi-ada-002",             38.05, 15.10, None),
    ("voyage-l2-instruct",        37.83, 10.85, None),
    ("VSS (ada-002)",             36.00, 12.63, None),
    ("BM25",                      31.25, 12.75, None),
    ("ColBERTv2",                 25.04, 11.75, None),
]

THEMES = {
    "light": dict(accent="#2a78d6", accent2="#8fb9e8", base="#c9c8c0",
                  ink="#24292f", ink2="#57606a", muted="#6e7781",
                  grid="#d8dee4", axis="#afb8c1"),
    "dark":  dict(accent="#3987e5", accent2="#1e5288", base="#484f58",
                  ink="#e6edf3", ink2="#9198a1", muted="#8b949e",
                  grid="#30363d", axis="#484f58"),
}

FONT = 'font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif"'
BAND, BAR, R = 30, 20, 4
TOP = 46
PLOT_H = len(ROWS) * BAND


def bar_path(x, y, w):
    if w <= R:
        return f'<rect x="{x}" y="{y}" width="{max(w,1):.1f}" height="{BAR}"/>'
    return (f'M {x} {y} h {w - R:.2f} a {R} {R} 0 0 1 {R} {R} v {BAR - 2 * R} '
            f'a {R} {R} 0 0 1 -{R} {R} h -{w - R:.2f} z')


def panel(x0, title, idx, maxv, ticks, t, label_w):
    plot_w_total = 0  # computed by caller convention; see PANEL_PLOT_W
    px = x0 + label_w
    out = [f'<text x="{px}" y="{TOP - 16}" {FONT} font-size="13" font-weight="600" fill="{t["ink"]}">{title}</text>']
    for tk in ticks:
        gx = px + PANEL_PLOT_W * tk / maxv
        out.append(f'<line x1="{gx:.1f}" y1="{TOP}" x2="{gx:.1f}" y2="{TOP + PLOT_H}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{gx:.1f}" y="{TOP + PLOT_H + 16}" {FONT} font-size="10.5" fill="{t["muted"]}" text-anchor="middle">{tk}</text>')
    out.append(f'<line x1="{px}" y1="{TOP}" x2="{px}" y2="{TOP + PLOT_H}" stroke="{t["axis"]}" stroke-width="1"/>')
    for i, (label, r20, h1, ours) in enumerate(ROWS):
        v = (r20, h1)[idx]
        y = TOP + i * BAND + (BAND - BAR) / 2
        w = PANEL_PLOT_W * v / maxv
        fill = t["accent"] if ours == "ours1" else t["accent2"] if ours == "ours2" else t["base"]
        out.append(f'<path d="{bar_path(px, y, w)}" fill="{fill}"/>')
        wt = ' font-weight="600"' if ours else ''
        ink = t["ink"] if ours else t["ink2"]
        if label_w:
            out.append(f'<text x="{px - 8}" y="{y + BAR - 5}" {FONT} font-size="12"{wt} fill="{ink}" text-anchor="end">{label}</text>')
        out.append(f'<text x="{px + w + 6:.1f}" y="{y + BAR - 5}" {FONT} font-size="11.5"{wt} fill="{ink}">{v:.1f}</text>')
    return "\n".join(out)


LABEL_W = 186
PANEL_PLOT_W = 330
GAP = 46
W = 10 + LABEL_W + PANEL_PLOT_W + 40 + GAP + PANEL_PLOT_W + 44
H = TOP + PLOT_H + 30

outdir = sys.argv[1] if len(sys.argv) > 1 else "."
for mode, t in THEMES.items():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="STARK-Prime test split: our evolved search vs leaderboard baselines on Recall at 20 and Hit at 1">']
    parts.append(panel(10, "Recall@20 (%)", 0, 60, [0, 20, 40, 60], t, LABEL_W))
    x2 = 10 + LABEL_W + PANEL_PLOT_W + 40 + GAP  # second panel shares row labels
    parts.append(panel(x2, "Hit@1 (%)", 1, 30, [0, 10, 20, 30], t, 0))
    parts.append("</svg>")
    path = os.path.join(outdir, f"stark-prime-leaderboard-{mode}.svg")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print("wrote", path, f"({W}x{H})")
