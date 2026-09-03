# Visual Sufficiency Framework (VSF)

**VSF (Version 2.2 "Certified Centers")** is an information-theoretic system for Exploratory Data Analysis (EDA). Given a dataset and a user-selected target value, VSF independently discovers the best feature subset for each dimensionality from 1D to 4D and reports **which cells of the resulting grid are certifiably almost purely that value, and how much of the value they capture.**

The headline is a triple — **coverage, number of certified centers, pooled purity** — not an association percentage. The distinction is the whole point of version 2.2 and is reproducible in three lines:

```python
import pandas as pd, vsf
df = pd.read_csv("data/adult_census.csv")
z  = vsf.binarize_target(df["occupation"].values, "Armed-Forces")   # 9 of 32 561 rows
codes, C = vsf.metrics.cell_codes(df[["workclass", "sex", "income"]].astype(str).values)
r = vsf.center_report(z, codes, C)
# U_adj = 41.3 %  ->  "an association exists"
# r.n_centers = 0, r.coverage = 0.0, max cell purity 2.42 % (8 of 330)
#             ->  "nothing on this display can be acted on"
```

Version 2.1 printed the 41.3 % as the headline. Version 2.2 prints `coverage 0 %, 0 centres` and says why.

---

## 🎯 Core Concept

* **Base Visualization (1D):** An axis represents a discretized feature interval. Inside each interval lies a circle:
  * **Circle Size:** The number of data points in that category (scaled by area, with the maximum circle filling the interval).
  * **Circle Color:** A discrete **3-zone** scale — no yellow — whose two boundaries belong to the user:
    * 🟢 **Discrete centre** — share of the target value $\ge \tau$ (and cell size $\ge$ `min_samples`). This is the deliverable, and it is **exactly the set `coverage` is computed from**: what is green on screen and what is counted are the same cells by construction.
    * 🟤 **Mixed** — between the lower boundary and $\tau$.
    * 🔴 **Low** — below the lower boundary.

    The two boundaries are different kinds of control and the interface separates them. $\tau$ *is* the definition of a centre, so moving it recomputes coverage, the centre count and the cross-validated figure. The lower boundary is purely cosmetic and re-colours instantly. $\tau = 100\,\%$ is allowed and means "only cells that are entirely the target value".

    Every cell also carries a Clopper–Pearson interval in its hover text. With `min_samples = 1` (the default) a cell holding one target-value object is a 100 %-pure centre; its interval reads $[\alpha/C,\ 100\,\%]$ — with $\alpha = 0.05$ over a 32-cell grid, $[0.16\,\%,\ 100\,\%]$ — so the weakness is visible on the cell rather than hidden by a threshold someone else picked. Two remedies are one control away — raise `min_samples`, or switch on strict mode, where a cell becomes a centre only when its purity reaches $\tau$ with simultaneous confidence $1-\alpha$ (exact binomial test, Bonferroni over every occupied cell). On pure noise at the worst grid density the framework permits, the default costs 2 spurious centres and 0.09 % coverage at $p = 0.23$; both remedies drive it to exactly zero.

* **Scaling to 2D, 3D, and 4D:** Adding features turns the line into a 2D planar grid, then into a 3D spatial cube. The 4th dimension (4D) is rendered as a stack of parallel frames (a film strip metaphor with an interactive slice stepper and auto-play controls).
* **Independent Branch Discovery (IBD):** The algorithm tests all feature combinations from 1D to 4D, ranking them by **bias-corrected** Mutual Information $\hat I - \mathbb{E}_0[\hat I]$. Due to feature synergy, the optimal 3D subset does not have to contain the optimal 2D subset. The system preserves the top-performing subset for each dimensionality separately — generating up to 4 independent "development branches".
* **Centres and coverage (new in 2.2):** VSF reports **coverage** (the share of all target-value objects inside green centres — the primary objective, driven toward 1), **$K$** (how many centres; at equal coverage fewer is better, which is the secondary objective and the search's tie-break) and **pooled purity**, each with a confidence bound, plus a **cross-validated** coverage with the Nadeau–Bengio corrected standard error. The 1D–4D search maximises coverage by default, so the branches it returns are the ones that best serve those two rules rather than the ones with the strongest statistical association. The cross-validated figure is the one that can *decrease* when a branch is over-resolved, which is what makes "how many characteristics describe this value?" a decidable question — `vsf.select_branch_dimensionality` answers it, and answers **None** when nothing certifies at any dimensionality. See `Project_Master_Document.md` Sections 0-ter, 3.3 and 5.3.
* **Bias correction and significance (2.1, retained as a diagnostic):** the plug-in MI estimator is positively biased by an amount that grows with the number of occupied grid cells, so on a dataset where *every* feature is independent of the target, ranking by raw MI returns branches reporting up to 9.8 % "normalized MI". VSF 2.1 subtracts the exact permutation expectation (Vinh, Epps & Bailey 2010) before ranking, reports $U_{\mathrm{adj}}$ instead of NMI, and attaches a permutation $p$-value — with an optional look-elsewhere-corrected variant for the whole 1D–4D search. See `Project_Master_Document.md` Sections 0-bis, 3.2 and 4.5.
* **Collapse/Split Animation (Object Constancy):** Within any selected branch, the user can smoothly collapse dimensions ($3D \to 2D \to 1D$) or split them back ($1D \to 2D \to 3D$). Objects merge and divide while strictly preserving relative geometric mass and relative sizes (Object Constancy).

---

## 📁 Project Structure

* `vsf/` — Installable Python library (`pip install -e .`):
  * `vsf/pmd.py` — Perceptually-Aligned Discretization (PMD).
  * `vsf/math.py` — Plug-in information-theoretic primitives (Shannon entropy, joint entropy, MI). Uncorrected by design; not for reporting.
  * `vsf/metrics.py` — Bias-corrected estimator layer: exact null expectation, adjusted MI, $U_{\mathrm{adj}}$, permutation and familywise nulls, per-class decomposition, Benjamini–Hochberg.
  * `vsf/centers.py` — Certified discrete centres: Clopper–Pearson/Wilson bounds without `scipy`, Bonferroni certification, coverage/$K$/purity, cross-validated coverage, exact multivariate-hypergeometric and familywise nulls, dimensionality selection.
  * `vsf/avr.py` — Independent Branch Discovery engine (1D–4D exhaustive search), with `objective="mi_adj"` (default) or `objective="coverage"`.
  * `vsf/vis.py` — Visualization payload generator for discrete centers rendering.
  * `vsf/server.py` — Embedded web server and REST API (entry point: `vsf.serve(df)`).
  * `vsf/dashboard.py` — Standalone single-file HTML exporter (`vsf.export_full_dashboard`).
* `vsf/webapp/` — Packaged web application assets (`index.html`, `static/css/styles.css`, `static/js/app.js`), served via `importlib.resources`.
* `run.py` — Interactive console launcher for benchmark datasets.
* `data/` — Core demo datasets (`mushrooms.csv`, `adult_census.csv`).
* `tests/` — Automated pytest regression test suite.
* `pyproject.toml` — Single source of truth for packaging and dependencies.

---

## 🚀 Quick Start

### Requirements
* Python **3.10+**
* Dependencies: `numpy`, `pandas` (declared in `pyproject.toml`).

### Installation
Install the package in editable mode:
```bash
pip install -e .
# or include test dependencies:
pip install -e ".[test]"
```

### Launching the Interactive Menu
Run the dataset launcher from the repository root:
```bash
python run.py
```
Or specify the dataset ID directly (e.g., `python run.py 4` to launch UCI Nursery).

### Using the Library with Custom Data
Call `vsf.serve(df)` in your Python code to automatically start the interactive visualizer for any `pandas.DataFrame`:
```python
import pandas as pd
import vsf

df = pd.read_csv("data/mushrooms.csv")
vsf.serve(df, host="127.0.0.1", port=8000)
```

### Standalone Offline HTML Export (`vsf.export_full_dashboard`)
Package the entire interactive web interface for a selected target into a **single self-contained HTML file** that opens locally in any browser without a running Python backend:
```python
import pandas as pd
import vsf

df = pd.read_csv("data/mushrooms.csv")
html = vsf.export_full_dashboard(df, target="class", criterion="p")
# A static export cannot be re-certified after the fact, so the certificate is
# baked in and stated in the exported page's own legend:
#   html = vsf.export_full_dashboard(
#       df, target="class", criterion="p",
#       center_spec=vsf.CenterSpec(tau=1.0, min_samples=10),   # only fully pure cells of >= 10 objects
#   )

with open("vsf_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
```

---

## 📖 Documentation

* Comprehensive scientific and mathematical specification: `Project_Master_Document.md`.
  * Section 0-ter — what version 2.2 changed and the reproducible case that forced it.
  * Section 3.3 — formal definitions: certified centre, coverage, cross-validated coverage, smallest sufficient dimensionality, and the two permutation nulls.
  * Section 5.3 — the certificate colour scale.
  * Section 10 — open problems, including the ones version 2.2 does *not* close.

---

## 📄 License

MIT — see the `LICENSE` file.
