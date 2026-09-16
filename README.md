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
    * 🟢 **Discrete centre** — by default, a cell **certified** to hold more than $\tau$ of the target value, with a guarantee that holds for the branch the search chose: an exact binomial test at $\alpha / T$, $T$ being every cell of every partition the search could show (Section 4.14). *Colouring* offers two legacy alternatives: the per-schema certificate (optimistic after a search) and the observed share $\ge \tau$ (no certificate). Whatever the choice, green cells are **exactly the set `coverage` is computed from**: what is green on screen and what is counted are the same cells by construction.
    * 🟤 **Mixed** — between the lower boundary and $\tau$.
    * 🔴 **Low** — below the lower boundary.

    The two boundaries are different kinds of control and the interface separates them. $\tau$ *is* the definition of a centre, so moving it recomputes coverage, the centre count and the cross-validated figure. The lower boundary is purely cosmetic and re-colours instantly. Under a certificate $\tau$ stops at 99 % (exact purity cannot be certified); with the observed-share colouring $\tau = 100\,\%$ means "only cells that are entirely the target value".

    Every cell also carries a Clopper–Pearson interval in its hover text. With `min_samples = 1` (the default) a cell holding one target-value object is a 100 %-pure centre; its interval reads $[\alpha/C,\ 100\,\%]$ — with $\alpha = 0.05$ over a 32-cell grid, $[0.16\,\%,\ 100\,\%]$ — so the weakness is visible on the cell rather than hidden by a threshold someone else picked. Two remedies are one control away — raise `min_samples`, or switch on strict mode, where a cell becomes a centre only when its purity reaches $\tau$ with simultaneous confidence $1-\alpha$ (exact binomial test, Bonferroni over every occupied cell). On pure noise at the worst grid density the framework permits, the default costs 2 spurious centres and 0.09 % coverage at $p = 0.23$; both remedies drive it to exactly zero.

* **Scaling to 2D, 3D, and 4D:** Adding features turns the line into a 2D planar grid, then into a 3D spatial cube. The 4th dimension (4D) is rendered as a stack of parallel frames (a film strip metaphor with an interactive slice stepper and auto-play controls).
* **Presence or absence:** every search has a direction. *Presence* certifies cells that are at least $\tau$ pure in the chosen value and reports how much of it they capture; *absence* certifies cells that are at least $\tau$ pure in its **complement** — where the value is almost missing — and reports how much of the data is certified free of it (drawn red, never green). Same search, same certificate, on the inverted indicator: $\text{absence}(v) \equiv \text{presence}(\lnot v)$. A threshold is accepted only above the base rate of the indicator searched: for a value that fills 90 % of the rows, presence needs $\tau > 0.90$ while absence is measured against $1 - p_0 = 10\,\%$ — the server refuses a void threshold with the reason rather than showing a uniformly green display (`direction=` in `discover_branches`, `/api/analyze`, `/api/scan/start`, `export_full_dashboard`; Project_Master_Document.md Section 4.8).
* **Solution landscape:** the search scores every candidate schema and keeps what it learned. A *Landscape* toggle replaces the lattice with a 10 × 10 lattice of (coverage, centres) categories — coverage in (0, 10], …, (90, 100] %, centres as a share of the largest count among the schemas shown — coloured by how many schemas fall in each cell under a three-zone scale you move; schemas certifying nothing are counted, not drawn; the winner is marked. Click a cell to list its schemas (most concentrated first, paged) and open any of them as a lattice (`/api/landscape`, `/api/landscape/cell`, `/api/analyze` with `features=`; `vsf.compute_landscape`, `vsf.report_schema`). An opened schema reports no uncorrected p-value — it was chosen by looking — but its cross-validated coverage stays out-of-sample. Section 4.9.
* **τ-curves:** in the Trade-offs view (Section 4.13) the upper panel shows, for each dimensionality, the best coverage any schema reaches at every whole-percent purity floor from the base rate to 100 % — four lines, the envelope of the landscape over τ, with the current boundary as a vertical cursor and the winning schema and its centre count in the hover. Where the line for d + 1 stops lying above the line for d, the extra axis has stopped paying for itself. Click a point to list the schemas of that d at that floor in the point's coverage category and open one at that floor (`/api/landscape/curves`, `/api/landscape/at`; `vsf.compute_tau_curves`, `vsf.tau_grid`). Section 4.9.
* **Trade-offs tab:** what coverage costs, in three panels. The τ-curves above, the number of certified centres the winning schema needs below them on the same purity-floor axis (the winner changes along the envelope — on `titanic` 15 times for 2D, 23 for 3D, 22 for 4D — so every change is ticked, and a *Track* control follows the selected branch instead), a band under both showing which dimensionality leads at each floor in that d's colour (ties to the smaller d; on `titanic` it breaks into 16 alternating stretches, which says the 3D and 4D envelopes run together rather than that the answer keeps changing), and, at the current floor, the Pareto staircase of coverage against the cost of the description. One control sets the cost unit for both panels: centres, or conditions ($d \times K$ — the (column = value) pairs actually written out), which is the only unit that puts the four dimensionalities on one scale, since a 4D rule is four times the text of a 1D one (on `titanic` the median envelope costs 1 / 7 / 29.5 / 45 centres, i.e. 1 / 14 / 88.5 / 180 conditions). The staircase is the part the branch list cannot show: on `titanic` at $\tau = 0.9$ the reported 2D winner certifies 49.7 % with 7 centres while `passenger_class + sex` certifies 47.1 % with **2** — 95 % of the coverage for under a third of the description. Maxima over a family chosen by looking, so optimistic and without an uncorrected p-value; the ratio coverage / centres is deliberately never computed, since it rewards degenerate one-centre schemas (`/api/landscape/frontier`, `/api/landscape/curves`; `vsf.avr.Landscape.frontier`). Section 4.13.
* **Data screen:** before any target is chosen, the guide screen reports what each column is (categories, largest category, single-row categories, missing rows, and flags for constant / more categories than the grid capacity / one category covering almost everything / a category *named* like a placeholder such as `unknown_deck`) and every pair of columns one of which (nearly) determines the other, measured as the share of rows the best deterministic map gets right, with the exact number of exception rows (on `titanic`: `title` fixes `sex` on 888 of 891 rows — an exact-duplicate check finds nothing there). Excluding a column travels as `drop=[...]` with every request, enters the analysis cache key and is echoed in the response; `prune=true` additionally skips candidate schemas that an exact dependency makes renamings of a smaller one. Once a target is named the same panel reports how well each single column fixes the target — the leakage check. Nothing is applied automatically (`/api/screen`; `vsf.screen_dataset`, `vsf.target_report`). Section 4.12.
* **Post-selection check:** a per-schema certificate holds for a schema fixed in advance, not for the one the search reports — on pure noise at least one reported branch carries a "certified" centre in 68 % of runs; that is why the default colouring is corrected over the whole search family. `vsf.certify_discovery` gives certificates that hold for the reported schemas (Bonferroni over every cell of every scored schema, or a search half and an evaluation half), and `vsf.nested_crossvalidation` gives the out-of-sample coverage of the whole search with the stability of the chosen schema. In the app: Branches → *Check this result* (`/api/validate`, background job); the static export carries the family certificate but not the check. Section 4.14.
* **Redundancy tab:** a third view beside Lattice and Landscape. It opens on the selected branch: for each of its centres, every centre of another scored schema that holds (almost) the same rows — *each* of the two holds at least $t$ of the other's rows (mutual containment, $|A \cap B| / \max(|A|, |B|) \ge t$, threshold yours, 50–100 %) — and the simplest such description when it needs fewer characteristics. A toggle shows *All centres* instead: every centre of every scored schema grouped around a representative (fewest characteristics, then the highest certified purity bound). Each list gives both one-sided shares, the overlap expected by chance and what each side holds that the other does not; the distribution the threshold cuts is drawn three times over the same axis — for the whole family and for the selected branch (each centre against its *nearest* other centre, above the cards), and inside each branch card for that one centre against *every* other centre, where the bars at or above the dashed line are exactly its listed alternatives — so the cut can be judged per case and not only family-wide (on `titanic` no family-wide cut exists). Every alternative is labelled with two separate facts — how its *characteristics* stand to the reference's (extends / shortens / neither) and how its *rows* do (=, ⊂, ⊃, crossing) — and a filter keeps only one kind; the row relation is measured, not inferred, because capacity coarsening can leave a schema's cells outside the cells of its own sub-schema (11.9 % of the nested schema pairs on `titanic`), and such pairs are marked rather than called sub-cells. Descriptive only: no centre, coverage or p-value changes (`/api/centers/branch`, `/api/centers/groups`, `/api/centers/group`; `vsf.collect_centers`). Section 4.11.
* **Composite targets:** the target can be a conjunction of two or three (column, value) pairs — *survived* ∧ *sex = female* — built from the catalog with the “+” beside a value. The indicator searched is the AND of the pairs, so nothing in the search changes; every column of the target leaves the feature space (its own columns would otherwise be found as the schema), and the target panel reports the rows, share and remaining features before the search runs (`/api/analyze` with `also=[[column, value], …]`, `/api/target`). No automatic enumeration of conjunctions: the user names the target. Section 4.10.
* **Independent Branch Discovery (IBD):** The algorithm tests all feature-column combinations from 1D to 4D — the category encoding of each column is fixed beforehand and independent of the target, so only which columns to combine is searched — and ranks them by **coverage of the chosen target value inside certified discrete centres**, tie-breaking on fewer centres, then more mass, then the best certified lower bound. Due to feature synergy, the optimal 3D subset does not have to contain the optimal 2D subset. The system preserves the top-performing subset for each dimensionality separately — generating up to 4 independent "development branches".
* **Centres and coverage:** VSF reports **coverage** (the share of all target-value objects inside green centres — the primary objective, driven toward 1), **$K$** (how many centres; at equal coverage fewer is better, which is the secondary objective and the search's tie-break) and **pooled purity**, each with a confidence bound, plus a **cross-validated** coverage with the Nadeau–Bengio corrected standard error. The cross-validated figure is the one that can *decrease* when a branch is over-resolved, which is what makes "how many characteristics describe this value?" a decidable question — `vsf.select_branch_dimensionality` answers it, and answers **None** when nothing certifies at any dimensionality.
* **No visualization without a chosen target:** a target value must always be resolvable before any search runs. On first load, or whenever no target/value is selected, the app shows a short guide instead of analyzing a default column — there is no meaningful branch to rank without a class of interest.
* **Collapse/Split Animation (Object Constancy):** Within any selected branch, the user can smoothly collapse dimensions ($3D \to 2D \to 1D$) or split them back ($1D \to 2D \to 3D$). Objects merge and divide while strictly preserving relative geometric mass and relative sizes (Object Constancy).

---

## 📁 Project Structure

* `vsf/` — Installable Python library (`pip install -e .`):
  * `vsf/pmd.py` — Category encoding (one category per distinct value) and grid capacity: at most ⌊N/10⌋ *occupied* cells per candidate, enforced by merging the rarest levels of nominal columns and adjacent values of numeric ones (Section 2.3).
  * `vsf/metrics.py` — Dense joint-cell codes (`cell_codes`, `dense_codes_from_flat`) plus Benjamini–Hochberg FDR control for the Global Pattern Scan.
  * `vsf/centers.py` — Certified discrete centres: Clopper–Pearson/Wilson bounds without `scipy`, Bonferroni certification, coverage/$K$/purity, cross-validated coverage, exact multivariate-hypergeometric and familywise nulls, dimensionality selection, and `coverage_score` (the branch-ranking key itself).
  * `vsf/avr.py` — Independent Branch Discovery engine (1D–4D exhaustive search over feature columns), always ranked by `coverage_score` against a resolvable positive class. `discover_branches_by_value` runs the same search for every value of one target column in a single enumeration (used by the Global Pattern Scan); the search internals (`_CandidateFactory`, `_exhaustive_search`) are exact re-implementations pinned by `tests/test_fastpaths.py`, see Project_Master_Document.md Section 4.6.
  * `vsf/screen.py` — Dataset screen: column profiles, (approximate) functional dependencies between columns, target leakage (Section 4.12).
  * `vsf/redundancy.py` — Redundant centres: centre catalogue of every scored schema, mutual-containment graph and leader grouping (Section 4.11).
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
