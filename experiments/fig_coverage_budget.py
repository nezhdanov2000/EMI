"""Figure: held-out coverage against the condition budget on four tables (paper Fig. 2)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = pd.read_csv(ROOT / "experiments/results/comparison_summary.csv")
panels = [("nursery", 0.7, "priority", "purity", 1, "nursery, priority, tau=0.7"),
          ("chess_krkp", 0.9, "won", "certified", 1, "chess, tau=0.9, certified"),
          ("connect_4", 0.9, "win", "certified", 20, "connect_4, win, tau=0.9, certified"),
          ("adult", 0.7, ">50K", "certified", 20, "adult, tau=0.7, certified")]
methods = [("vsf", "full grid", "#555555", "-"), ("vsf_partial", "partial centres", "#c0392b", "-"),
           ("rules_disjoint", "free rules (disjoint)", "#2874a6", "--"), ("ssdpp", "SSD++", "#1e8449", ":"),
           ("tree", "CART", "#b7950b", "-.")]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
for ax, (ds, tau, pos, sel, m, title) in zip(axes, panels):
    g = s[(s.dataset == ds) & (s.tau == tau) & (s.positive == pos) & (s.selection == sel) & (s.min_samples == m)]
    for meth, lab, col, ls in methods:
        q = g[g.method == meth].sort_values("budget")
        if len(q):
            ax.errorbar(q.budget, 100 * q.coverage, yerr=100 * q.coverage_se, label=lab, color=col, ls=ls,
                        marker="o", ms=3, lw=1.4, capsize=2)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax.set_title(title, fontsize=9); ax.set_xlabel("conditions B", fontsize=9); ax.grid(alpha=.3)
axes[0].set_ylabel("held-out coverage, %", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(ROOT / "paper/fig_coverage_budget.pdf")
print("wrote paper/fig_coverage_budget.pdf")
