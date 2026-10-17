#!/usr/bin/env python3
"""Write the README banner (docs/img/banner-{light,dark}.svg).

Every number on the banner is a committed result; the source of each is next to it below.
Change a number here only after the result it cites has changed.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"

STEPS = [  # (title, subtitle)
    ("CUDA", "your kernels"),
    ("migrate", "SYCLomatic"),
    ("compile", "icpx -fsycl"),
    ("scan_helpers", "0 dpct left"),
    ("verify", "vs reference"),
    ("GPU check", "NVIDIA · AMD"),
]

STATS = [  # (big, small, caption lines, source)
    ("40/40", "vs 23/40", ["correct & helper-free, with vs", "without the toolkit (4 workloads)"],
     "docs/ablation/round2"),
    ("30/30", "vs 18/30", ["on 5 HeCBench programs", "the agent had never seen"],
     "docs/agent-ports"),
    ("10/10", "PASS", ["agent-written ports, unedited,", "on NVIDIA and AMD MI350X GPUs"],
     "docs/agent-ports"),
]

PALETTES = {
    "light": dict(bg="#ffffff", border="#d0d7de", ink="#1f2328", muted="#59636e",
                  box="#f6f8fa", box_border="#d0d7de", accent="#0969da", accent_soft="#ddf4ff",
                  good="#1a7f37", arrow="#8c959f"),
    "dark": dict(bg="#0d1117", border="#30363d", ink="#e6edf3", muted="#9198a1",
                 box="#161b22", box_border="#30363d", accent="#4493f8", accent_soft="#121d2f",
                 good="#3fb950", arrow="#6e7681"),
}

W, H = 1200, 500
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, "
        "sans-serif")
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"


def text(x, y, s, size, fill, weight=400, anchor="start", family=FONT):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def banner(p: dict[str, str]) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" role="img" aria-label="oneapi-agent-toolkit: evidence-first '
           'tools that help coding agents finish CUDA to SYCL ports and prove them">',
           f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="{p["bg"]}" '
           f'stroke="{p["border"]}"/>']
    # title
    out.append(text(48, 74, "oneapi-agent-toolkit", 40, p["ink"], 700))
    out.append(text(48, 108, "Evidence-first tools that help coding agents finish CUDA → SYCL "
                    "ports, and prove them.", 20, p["muted"]))

    # pipeline
    n, x0, y0, bw, bh, gap = len(STEPS), 48, 146, 160, 74, 28
    for i, (title, sub) in enumerate(STEPS):
        x = x0 + i * (bw + gap)
        first, last = i == 0, i == n - 1
        fill = p["accent_soft"] if not first else p["box"]
        stroke = p["accent"] if not first else p["box_border"]
        out.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="10" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="1.5"/>')
        fam = FONT if first or last else MONO
        out.append(text(x + bw / 2, y0 + 33, title, 18, p["ink"], 600, "middle", fam))
        out.append(text(x + bw / 2, y0 + 56, sub, 14, p["muted"], 400, "middle"))
        if not last:
            ax = x + bw + 4
            out.append(f'<path d="M{ax} {y0 + bh / 2} h{gap - 10}" stroke="{p["arrow"]}" '
                       'stroke-width="2" fill="none"/>')
            out.append(f'<path d="M{ax + gap - 10} {y0 + bh / 2 - 5} l7 5 l-7 5 z" '
                       f'fill="{p["arrow"]}"/>')
    out.append(text(48, y0 + bh + 34, "Done only when verify passes and scan_helpers finds "
                    "zero dpct helpers, on the same freshly compiled source.", 15, p["muted"]))

    # stat cards
    cw, ch, cy, cgap = 352, 150, 290, 24
    for i, (big, small, caption, _src) in enumerate(STATS):
        x = 48 + i * (cw + cgap)
        out.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" rx="12" '
                   f'fill="{p["box"]}" stroke="{p["box_border"]}"/>')
        out.append(f'<text x="{x + 24}" y="{cy + 58}" font-family="{FONT}">'
                   f'<tspan font-size="44" font-weight="700" fill="{p["good"]}">{escape(big)}</tspan>'
                   f'<tspan dx="12" font-size="20" font-weight="600" fill="{p["muted"]}">'
                   f'{escape(small)}</tspan></text>')
        for j, line in enumerate(caption):
            out.append(text(x + 24, cy + 94 + j * 22, line, 16, p["ink"]))
    out.append(text(48, 476, "Claude Sonnet 5.5 and Haiku 5.5 · every port graded by a hidden "
                    "check the agent never saw · each number links to its evidence below", 14,
                    p["muted"]))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pal in PALETTES.items():
        (OUT / f"banner-{name}.svg").write_text(banner(pal))
        print(OUT / f"banner-{name}.svg")


if __name__ == "__main__":
    main()
