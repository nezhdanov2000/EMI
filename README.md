# Visual Sufficiency Framework (VSF)

**VSF** is an Exploratory Data Analysis (EDA) tool for **categorical data**. Given a dataset and a user-chosen target value, VSF independently discovers the best feature subset for each dimensionality from 1D to 4D and reports **which cells of the resulting grid are certifiably almost purely that value, and how much of the value they capture.**

The headline is a triple — **coverage, number of certified centers, pooled purity**. There is no other ranking mode: every branch search requires a resolvable positive class (an explicit target value, or a target column that is itself two-valued) and always ranks candidates by coverage of that class, reproducible in three lines:

```python
import pandas as pd, vsf

# UCI "Adult" / Census Income dataset (https://archive.ics.uci.edu/dataset/2/adult)
# -- a public dataset, not bundled with this library.
df = pd.read_csv("adult_census.csv")
z  = vsf.binarize_target(df["occupation"].values, "Armed-Forces")   # 9 of 32 561 rows
codes, C = vsf.metrics.cell_codes(df[["workclass", "sex", "income"]].astype(str).values)
r = vsf.center_report(z, codes, C)
# r.n_centers = 0, r.coverage = 0.0, max cell purity 2.42 % (8 of 330)
#             ->  "nothing on this display can be acted on"
```

**Everything is a category.** Every column — feature or target — is encoded by its distinct values, whatever the dtype: `1`, `2`, `3` are three categories exactly as `low`, `mid`, `high` would be. There is no continuous-feature model: no binning, no bin-count heuristic, no cut points, and **no information-theoretic quantity computed anywhere in the package**. Everything reported is a count ratio with an exact binomial bound. A genuinely continuous column is encoded as it stands — thousands of distinct floats become thousands of categories — rather than being silently quantized; preparing such a column is the caller's decision, and the framework makes it visible instead of hiding it behind a density heuristic. See `Project_Master_Document.md` Section 2 for what was removed with the old binning layer and why.

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
* **Independent Branch Discovery (IBD):** The algorithm tests all feature-column combinations from 1D to 4D — the category encoding of each column is fixed beforehand and independent of the target, so only which columns to combine is searched — and ranks them by **coverage of the chosen target value inside certified discrete centres**, tie-breaking on fewer centres, then more mass, then the best certified lower bound. Due to feature synergy, the optimal 3D subset does not have to contain the optimal 2D subset. The system preserves the top-performing subset for each dimensionality separately — generating up to 4 independent "development branches".
* **Centres and coverage:** VSF reports **coverage** (the share of all target-value objects inside green centres — the primary objective, driven toward 1), **$K$** (how many centres; at equal coverage fewer is better, which is the secondary objective and the search's tie-break) and **pooled purity**, each with a confidence bound, plus a **cross-validated** coverage with the Nadeau–Bengio corrected standard error. The cross-validated figure is the one that can *decrease* when a branch is over-resolved, which is what makes "how many characteristics describe this value?" a decidable question — `vsf.select_branch_dimensionality` answers it, and answers **None** when nothing certifies at any dimensionality.
* **No visualization without a chosen target:** a target value must always be resolvable before any search runs. On first load, or whenever no target/value is selected, the app shows a short guide instead of analyzing a default column — there is no meaningful branch to rank without a class of interest.
* **Collapse/Split Animation (Object Constancy):** Within any selected branch, the user can smoothly collapse dimensions ($3D \to 2D \to 1D$) or split them back ($1D \to 2D \to 3D$). Objects merge and divide while strictly preserving relative geometric mass and relative sizes (Object Constancy).

---

## 📁 Project Structure

* `vsf/` — Installable Python library (`pip install -e .`):
  * `vsf/pmd.py` — Category encoding (one category per distinct value) and grid-capacity level merging.
  * `vsf/metrics.py` — Dense joint-cell codes (`cell_codes`, `dense_codes_from_flat`) plus Benjamini–Hochberg FDR control for the Global Pattern Scan.
  * `vsf/centers.py` — Certified discrete centres: Clopper–Pearson/Wilson bounds without `scipy`, Bonferroni certification, coverage/$K$/purity, cross-validated coverage, exact multivariate-hypergeometric and familywise nulls, dimensionality selection, and `coverage_score` (the branch-ranking key itself).
  * `vsf/avr.py` — Independent Branch Discovery engine (1D–4D exhaustive search over feature columns), always ranked by `coverage_score` against a resolvable positive class. `discover_branches_by_value` runs the same search for every value of one target column in a single enumeration (used by the Global Pattern Scan); the search internals (`_CandidateFactory`, `_exhaustive_search`) are exact re-implementations pinned by `tests/test_fastpaths.py`, see Project_Master_Document.md Section 4.6.
  * `vsf/vis.py` — Visualization payload generator for discrete centers rendering.
  * `vsf/server.py` — Embedded web server and REST API (entry point: `vsf.serve(df)`).
  * `vsf/dashboard.py` — Standalone single-file HTML exporter (`vsf.export_full_dashboard`).
* `vsf/webapp/` — Packaged web application assets (`index.html`, `static/css/styles.css`, `static/js/app.js`), served via `importlib.resources`.
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

### Using the Library with Custom Data
Call `vsf.serve(df)` in your Python code to automatically start the interactive visualizer for any `pandas.DataFrame`:
```python
import pandas as pd
import vsf

df = pd.read_csv("your_data.csv")
vsf.serve(df, host="127.0.0.1", port=8000)
```
After each analysis the server precomputes the other values of the same target column in the background so the next clicks are instant (`prefetch=False` turns this off; the responses are byte-identical either way), and the page pre-triangulates every catalog label for WebGL while the guide is shown so the first render does not pay for it.

### Standalone Offline HTML Export (`vsf.export_full_dashboard`)
Package the entire interactive web interface for a selected target into a **single self-contained HTML file** that opens locally in any browser without a running Python backend:
```python
import pandas as pd
import vsf

df = pd.read_csv("your_data.csv")
html = vsf.export_full_dashboard(df, target="my_target_column", criterion="p")
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

* `Project_Master_Document.md` carries the full specification: the categorical data model, the certified-centres definitions and their guarantees, the branch-discovery algorithm and its cost, the display, and the open limitations.

---

## 📄 License

MIT — see the `LICENSE` file.
