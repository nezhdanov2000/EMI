"""
Figure 1 of the paper: the winning 3-D schema on nursery (class = priority,
tau = 0.7, m = 1, all rows) as the full grid and with partial-axis centres.

Panel (a): the cells of purity >= tau (full-grid centres), ringed. Panel (b):
the disjoint partial centres `vsf.partial.partial_centres_for_schema`
selects for the same schema: a bar (a cell of a two-column sub-grid) is a
line through its three member cells along the free axis; a full cell is
ringed. Spheres are the occupied cells, coloured by purity with a single-hue
sequential scale; nursery is a full factorial design, so every cell has the
same number of rows and size encodes nothing. Writes paper/fig_example.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from datasets import load_dataset  # noqa: E402
from vsf.avr import _prepare_search  # noqa: E402
from vsf.centers import CenterSpec  # noqa: E402
from vsf.partial import partial_centres_for_schema  # noqa: E402

DATASET, POSITIVE, TAU, M = "nursery", "priority", 0.7, 1
SCHEMA = ("parents", "has_nurs", "health")
HIGHLIGHT = "#d55e00"   # Okabe-Ito vermilion: the one accent, never a series colour


def main() -> int:
    frame, entry = load_dataset(DATASET)
    feats = [c for c in frame.columns if c != entry.target]
    X, Z = frame[feats].values, frame[entry.target].values
    spec = CenterSpec(tau=TAU, min_samples=M)
    schema = tuple(feats.index(c) for c in SCHEMA)
    sel = partial_centres_for_schema(X, Z, schema, spec, POSITIVE)
    prepared = _prepare_search(X, Z, feats, POSITIVE, spec, "presence")
    assert prepared is not None
    factory, zb, _ = prepared
    raw = factory._raw
    codes3, n3 = factory.codes(schema)
    cells: Dict[int, Tuple[List[int], int, int]] = {}
    for c in range(n3):
        rows = np.flatnonzero(codes3 == c)
        if rows.size:
            cells[c] = ([int(raw[rows[0], j]) for j in schema], int(rows.size), int(zb[rows].sum()))
    labels: Dict[str, List[str]] = {}
    for j in schema:
        m: Dict[int, str] = {}
        for code, s in zip(raw[:, j], X[:, j]):
            m.setdefault(int(code), str(s))
        labels[feats[j]] = [m[k] for k in sorted(m)]
    bars: List[List[int]] = []
    singles: List[int] = []
    for cn in sel.centres:
        codes_s, _ = factory.codes(tuple(cn.sub_schema))
        member = sorted({int(v) for v in codes3[np.flatnonzero(codes_s == cn.cell)]})
        (bars if len(member) > 1 else singles).append(member if len(member) > 1 else member[0])  # type: ignore[arg-type]
    full = [c for c, (_, n, k) in cells.items() if k / n >= TAU]
    full_cov = sum(cells[c][2] for c in full) / sel.n_positive
    cond = sum(cn.cost for cn in sel.centres)

    cmap, norm = plt.get_cmap("Blues"), colors.Normalize(0, 1)

    def panel(ax, title: str, ringed: Sequence[int], lines: Sequence[Sequence[int]]) -> None:
        pts = np.array([v for v, _, _ in cells.values()], dtype=float)
        pur = [k / n for _, n, k in cells.values()]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pur, cmap=cmap, norm=norm, s=120,
                   edgecolor="#555555", linewidth=0.6, depthshade=False)
        for c in ringed:
            v = cells[c][0]
            ax.scatter([v[0]], [v[1]], [v[2]], s=300, facecolors="none", edgecolors=HIGHLIGHT,
                       linewidths=2.2, depthshade=False)
        for cellset in lines:
            p = np.array([cells[c][0] for c in cellset], dtype=float)
            p = p[np.argsort(p[:, 0])]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=HIGHLIGHT, linewidth=3.2, solid_capstyle="round")
        for axis, name, kw in ((ax.xaxis, SCHEMA[0], dict(rotation=12, ha="right")),
                               (ax.yaxis, SCHEMA[1], {}), (ax.zaxis, SCHEMA[2], dict(ha="left"))):
            axis.set_ticks(range(len(labels[name])))
            axis.set_ticklabels(labels[name], fontsize=7.5, **kw)
        ax.set_xlabel(SCHEMA[0], fontsize=9, labelpad=6)
        ax.set_ylabel(SCHEMA[1], fontsize=9, labelpad=12)
        ax.set_zlabel(SCHEMA[2], fontsize=9, labelpad=24)
        ax.set_title(title, fontsize=9.5, y=0.96)
        ax.view_init(elev=20, azim=-62)
        ax.set_box_aspect((3, 5, 3), zoom=0.9)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)

    fig = plt.figure(figsize=(11.5, 5.2))
    ax1 = fig.add_axes([0.0, 0.0, 0.45, 0.9], projection="3d")
    ax2 = fig.add_axes([0.47, 0.0, 0.45, 0.9], projection="3d")
    panel(ax1, f"(a) full-grid centres\n{len(full)} cells, {3 * len(full)} conditions, coverage {100 * full_cov:.1f}%",
          full, [])
    panel(ax2, f"(b) partial-axis centres\n{len(bars)} bars + {len(singles)} cells, {cond} conditions, "
               f"coverage {100 * sel.k_sel / sel.n_positive:.1f}%", singles, bars)
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.5])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f"cell purity k/n (share of class = {POSITIVE})", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(TAU, color=HIGHLIGHT, linewidth=1.6)
    cb.ax.text(1.4, TAU, rf"$\tau={TAU}$", fontsize=7, va="center", color=HIGHLIGHT,
               transform=cb.ax.get_yaxis_transform())
    out = ROOT / "paper" / "fig_example.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}: full grid {len(full)} cells {100 * full_cov:.1f} %, partial {len(sel.centres)} centres "
          f"{cond} conditions {100 * sel.k_sel / sel.n_positive:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
