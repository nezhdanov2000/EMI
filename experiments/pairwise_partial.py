"""
Paired comparison of vsf_partial with vsf and with rules_disjoint from the
per-split cache of compare_baselines.py (PMD 4.16). Nadeau-Bengio SE;
a mark needs |t| >= 2. Uninformative configurations (no method reaches 1 %
held-out coverage) are left out. Writes results/pairwise_partial.csv.

Two kinds of mark are written for the three main comparisons. `<cmp>` is
the descriptive mark, |t| >= 2 on the Nadeau-Bengio t with J - 1 degrees
of freedom (J = number of splits): at 5 x 5 folds this is a two-sided
level of about 0.057, at 5 x 1 about 0.12. `<cmp>_holm` marks a cell only
when its two-sided p-value survives a Holm correction over the informative
budgets of the SAME configuration (at most six tests) at level 0.05, so
that a configuration cannot contribute a win at every budget from one
lucky split. Neither correction spans configurations: the counts in the
paper are counts of cells, not a test of a global hypothesis.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_baselines import CONFIGS, N_SPLITS, RESULTS, BUDGETS, _cache_path, CACHE_EXTRA  # noqa: E402
from datasets import load_dataset  # noqa: E402


def nb_t(diff: np.ndarray, rho: float) -> float:
    J = len(diff)
    var = diff.var(ddof=1) * (1.0 / J + rho) if J > 1 else 0.0
    return float(diff.mean() / np.sqrt(var)) if var > 0 else 0.0


def mark(t: float) -> str:
    return "+" if t >= 2 else ("-" if t <= -2 else "=")


def p_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of a Nadeau-Bengio t with df degrees of freedom (1.0 when df < 1)."""
    if df < 1 or not np.isfinite(t):
        return 1.0
    return float(2.0 * student_t.sf(abs(t), df))


def holm_marks(ts: List[float], ps: List[float], alpha: float = 0.05) -> List[str]:
    """
    Holm step-down over one family of tests: '+' / '-' by the sign of t where
    the adjusted p-value is <= alpha, '=' otherwise.
    """
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    adjusted = [1.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * ps[i])
        adjusted[i] = min(1.0, running)
    return ["+" if (adjusted[i] <= alpha and ts[i] > 0) else ("-" if (adjusted[i] <= alpha and ts[i] < 0) else "=")
            for i in range(n)]


HOLM_COMPARISONS = ("partial_vs_vsf", "partial_vs_rules_disjoint", "vsf_vs_rules_disjoint")


def main() -> int:
    rows: List[Dict[str, object]] = []
    frames: Dict[str, tuple] = {}
    for cfg in CONFIGS:
        if cfg.dataset not in frames:
            frame, entry = load_dataset(cfg.dataset)
            feats = [c for c in frame.columns if c != entry.target]
            frames[cfg.dataset] = (frame[feats].values, frame[entry.target].values)
        X, Z = frames[cfg.dataset]
        splits = []
        for i in range(N_SPLITS * cfg.n_repeats):
            p = _cache_path(cfg.dataset, cfg, X, Z, 0, i)
            if not p.exists():
                break
            methods = json.loads(p.read_text())["methods"]
            extra_p = CACHE_EXTRA / p.name
            if extra_p.exists():
                extra_methods = json.loads(extra_p.read_text()).get("methods", {})
                for k, v in extra_methods.items():
                    methods.setdefault(k, v)
            splits.append(methods)
        if len(splits) < N_SPLITS * cfg.n_repeats:
            print(f"{cfg.dataset} {cfg.tau} {cfg.selection}: incomplete, skipped")
            continue
        rho = 1.0 / (N_SPLITS - 1)
        df = len(splits) - 1
        config_rows: List[Dict[str, object]] = []
        for b in BUDGETS:
            present = [m for m in ("vsf", "vsf_partial", "vsf_greedy", "rules_disjoint", "rules", "tree", "ssdpp",
                                   "wracc", "wracc_raw", "tree_opt")
                       if all(m in s for s in splits)]
            cov = {m: np.array([s[m][str(b)]["k_in"] / max(1, s[m][str(b)]["n_pos"]) for s in splits])
                   for m in present}
            pur = {m: np.array([s[m][str(b)]["k_in"] / s[m][str(b)]["n_in"] if s[m][str(b)]["n_in"] else np.nan
                                for s in splits]) for m in cov}
            main_cov = [cov[m].mean() for m in ("vsf", "vsf_partial", "vsf_greedy", "rules_disjoint", "rules", "tree", "ssdpp") if m in cov]
            if not main_cov or max(main_cov) < 0.01:
                continue
            t_full = nb_t(cov["vsf_partial"] - cov["vsf"], rho)
            t_rules = nb_t(cov["vsf_partial"] - cov["rules_disjoint"], rho)
            t_grid_rules = nb_t(cov["vsf"] - cov["rules_disjoint"], rho)
            t_tree = nb_t(cov["vsf_partial"] - cov["tree"], rho)
            t_ssd = nb_t(cov["vsf_partial"] - cov["ssdpp"], rho)
            t_greedy = nb_t(cov["vsf"] - cov["vsf_greedy"], rho)
            # net coverage (k - tau/(1-tau)(n-k))/P and stability (mean pairwise Jaccard of held-out unions)
            net = {}
            stab = {}
            for m in cov:
                ks = np.array([s[m][str(b)]["k_in"] for s in splits], dtype=float)
                ns = np.array([s[m][str(b)]["n_in"] for s in splits], dtype=float)
                ps = np.array([max(1, s[m][str(b)]["n_pos"]) for s in splits], dtype=float)
                net[m] = float(((ks - cfg.tau / (1 - cfg.tau) * (ns - ks)) / ps).mean())
            t_net = nb_t(np.array([(s["vsf_partial"][str(b)]["k_in"] - cfg.tau / (1 - cfg.tau) * (s["vsf_partial"][str(b)]["n_in"] - s["vsf_partial"][str(b)]["k_in"])) / max(1, s["vsf_partial"][str(b)]["n_pos"])
                                    - (s["rules"][str(b)]["k_in"] - cfg.tau / (1 - cfg.tau) * (s["rules"][str(b)]["n_in"] - s["rules"][str(b)]["k_in"])) / max(1, s["rules"][str(b)]["n_pos"])
                                    for s in splits]), rho)
            # WRAcc top-k lists (EXTRA_METHODS): present only where the cache was topped up
            wr = {}
            for m in ("wracc", "wracc_raw", "tree_opt"):
                if m in cov:
                    wr[f"cov_{m}"] = cov[m].mean()
                    wr[f"partial_vs_{m}"] = mark(nb_t(cov["vsf_partial"] - cov[m], rho))
                    wr[f"purity_ok_{m}"] = np.nanmean(pur[m] >= cfg.tau)
                    wr[f"net_{m}"] = net[m]
                else:
                    wr[f"cov_{m}"] = np.nan
                    wr[f"partial_vs_{m}"] = ""
                    wr[f"purity_ok_{m}"] = np.nan
                    wr[f"net_{m}"] = np.nan
            wr["tree_opt_timeouts"] = (float(np.mean([any("timeout" in lab for lab in s["tree_opt"][str(b)]["labels"]) for s in splits]))
                                       if "tree_opt" in cov else np.nan)
            gap = cov["rules_disjoint"].mean() - cov["vsf"].mean()
            closed = (cov["vsf_partial"].mean() - cov["vsf"].mean()) / gap if gap >= 0.01 else np.nan
            config_rows.append(dict(
                dataset=cfg.dataset, positive=cfg.positive, tau=cfg.tau, m=cfg.min_samples,
                selection=cfg.selection, budget=b, n_splits=len(splits),
                t_partial_vs_vsf=t_full, t_partial_vs_rules_disjoint=t_rules, t_vsf_vs_rules_disjoint=t_grid_rules,
                p_partial_vs_vsf=p_two_sided(t_full, df), p_partial_vs_rules_disjoint=p_two_sided(t_rules, df),
                p_vsf_vs_rules_disjoint=p_two_sided(t_grid_rules, df),
                cov_vsf=cov["vsf"].mean(), cov_partial=cov["vsf_partial"].mean(),
                cov_rules_disjoint=cov["rules_disjoint"].mean(), cov_rules=cov["rules"].mean(), cov_tree=cov["tree"].mean(),
                cov_greedy=cov["vsf_greedy"].mean(),
                partial_vs_vsf=mark(t_full), partial_vs_rules_disjoint=mark(t_rules), vsf_vs_rules_disjoint=mark(t_grid_rules),
                partial_vs_tree=mark(t_tree), vsf_vs_greedy=mark(t_greedy),
                cov_ssdpp=cov["ssdpp"].mean(), partial_vs_ssdpp=mark(t_ssd),
                purity_ok_ssdpp=np.nanmean(pur["ssdpp"] >= cfg.tau),
                net_partial=net["vsf_partial"], net_rules=net["rules"], net_rules_disjoint=net["rules_disjoint"], net_vsf=net["vsf"], net_tree=net["tree"], net_ssdpp=net["ssdpp"],
                partial_vs_rules_net=mark(t_net),
                **wr,
                gap_closed=closed,
                purity_ok_vsf=np.nanmean(pur["vsf"] >= cfg.tau), purity_ok_partial=np.nanmean(pur["vsf_partial"] >= cfg.tau),
                purity_ok_rules_disjoint=np.nanmean(pur["rules_disjoint"] >= cfg.tau), purity_ok_rules=np.nanmean(pur["rules"] >= cfg.tau),
            ))
        # Holm over the informative budgets of this configuration, per comparison.
        for cmp_name in HOLM_COMPARISONS:
            marks = holm_marks([float(r[f"t_{cmp_name}"]) for r in config_rows],
                               [float(r[f"p_{cmp_name}"]) for r in config_rows])
            for r, hm in zip(config_rows, marks):
                r[f"{cmp_name}_holm"] = hm
        rows.extend(config_rows)
    d = pd.DataFrame(rows)
    out = RESULTS / "pairwise_partial.csv"
    d.to_csv(out, index=False)
    for sel, s in d.groupby("selection"):
        print(f"\n== selection = {sel}: {s.groupby(['dataset','tau']).ngroups} configs x budgets = {len(s)} cells")
        for col in ("partial_vs_vsf", "partial_vs_rules_disjoint", "vsf_vs_rules_disjoint", "partial_vs_tree", "vsf_vs_greedy", "partial_vs_rules_net",
                    "partial_vs_vsf_holm", "partial_vs_rules_disjoint_holm", "vsf_vs_rules_disjoint_holm"):
            print(f"  {col:33s}", s[col].value_counts().reindex(['+', '=', '-'], fill_value=0).to_dict())
        print("  mean net coverage at B=32:", {k: round(float(s[s.budget == 32][k].mean()), 3) for k in ("net_vsf", "net_partial", "net_rules_disjoint", "net_rules", "net_tree")})
        g = s.gap_closed.dropna()
        if len(g):
            print(f"  gap closed (rules >= grid + 1pp, n={len(g)}): median {g.median():.2f} mean {g.mean():.2f}")
        print("  share of splits with held-out purity >= tau:",
              {k: round(float(s[k].mean()), 3) for k in ("purity_ok_vsf", "purity_ok_partial", "purity_ok_rules_disjoint", "purity_ok_rules")})
        print(s[s.budget == 32][["dataset", "tau", "cov_vsf", "cov_partial", "cov_rules_disjoint", "cov_rules", "cov_tree",
                                 "partial_vs_vsf", "partial_vs_rules_disjoint"]].round(3).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
