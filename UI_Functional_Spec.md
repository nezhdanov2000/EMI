# VSF UI & Functional Specification

**Version 2.0 — aligned with `Project_Master_Document.md` v2.0 ("Clean Core").** Version 1.0 of this document described an interface around an Auto-Discovery catalog over all dataset columns and 7-channel visualization (1D–7D). Both ideas were removed from the core — see `Project_Master_Document.md`, Section 0. This document has been completely rewritten rather than updated piecemeal, because the old structure (Master-Detail: insights catalog → visualizer) was entirely built around the removed Auto-Discovery.

This document describes the user interface (UI) architecture and interaction logic for the Visual Sufficiency Framework (VSF) v2.0.

---

## 1. Interface Concept

The user explicitly specifies the target variable (column + optional criterion-value — see `Project_Master_Document.md` Section 1.4). The system runs Independent Branch Discovery (Section 4 of the master document) and presents **up to 4 independent branches** (1D, 2D, 3D, 4D). The interface is divided into two areas:

1. **Target and Branch Selector (Master):** selecting the target column/criterion and switching between the up to 4 found branches.
2. **Discrete Centers Visualizer (Detail):** the workspace where the grid of circles for the selected branch is rendered, with collapse/split animations within it.

**Explicitly missing compared to version 1.0:** 5D–7D modes with Hue/Saturation/Lightness encoding; mining "dirty centers" on click; composite AND-filter from multiple columns; transition to Graph Inference.

**Returned in a new form:** a way to see "which column is interesting at all" — requested explicitly after v2.0 as **Global Pattern Scan** (Section 2.1) — but NOT as v1.0's Auto-Discovery catalog (which automatically showed up by default and replaced standard target selection). Global Pattern Scan is an optional, on-demand background scan that only **filters** the already existing flat Target Selector (Section 2) by an NMI threshold; nothing is substituted automatically, significance is still not checked (the same principle of "raw MI/NMI, without proof statements", Section 4.5 of the master document).

---

## 2. Target Selector Section

* **Description:** a flat list of dataset columns (without One-vs-Rest propositionalization of the entire dataset — just showing the available columns and their observed values, exactly as `/api/columns` already returns).
* **Interaction:** the user selects a column as $Z$; optionally, a specific value of this column as a binary criterion. No precomputed `Max MI`/`Optimal d*` is shown at this level — they only appear after launching Independent Branch Discovery for the selected target (see Section 3).

## 2.1. Global Pattern Scan (Display Settings)

* **Location:** an "NMI Threshold (%)" input field + a "Scan Dataset" button in the Display Settings panel (not in the Target Selector itself — the scan filters its result, but is not part of it).
* **Scanning Logic** (live application only, `vsf.serve()` — the static export does not have a server for this computation):
  1. Each dataset column is taken.
  2. For each OBSERVED value of this column, a binary criterion Z = (column == value) is built (One-vs-Rest, the exact same form as a regular criterion in Section 2), and the same honest exhaustive 1D-4D search (Section 4 of the master doc) is run across all OTHER columns, just as for a single analysis.
  3. The maximum NMI among the found branches (1D..4D) for this value is taken. If it is strictly greater than the given threshold, the pair (column, value) is kept; otherwise, it is discarded.
  4. The "Target" section (Section 2) is rebuilt — keeping only columns that have at least one value passing the filter, and within them, only the passing values (marked with their max NMI and the dimensionality of the best branch).
* **Cost — honest and unembellished:** this is not an approximation or a separate cheaper algorithm — it is exactly the same exhaustive search as `/api/analyze`, just executed for EVERY (column, value) pair in the dataset. For a medium-sized dataset (~20 columns), this can take from a minute to several tens of minutes. Therefore, the scan is performed in a background thread on the server with a progress bar ("N out of M pairs") and a "Cancel" button — a non-blocking request. Reloading the page during a scan does not lose progress (the server keeps the scan state, the page just resumes polling the status).
* **Return to the full list:** a "Show All Columns/Values" button appears after the scan finishes/cancels and removes the filter without re-requesting the server (the list is already loaded, just the client-side filter is removed).
* **Explicitly NOT in scope:** static export (`export_full_dashboard`) — there is no server there to run a scan in the browser.

## 3. Branch Selector Section

* **Description:** after the target $Z$ is set, the system computes up to 4 branches (Section 4 of the master document) and shows them as a parallel list, rather than as a nested Pareto front:
  * 1D: $S^*_1$ → $I = \dots$
  * 2D: $S^*_2$ → $I = \dots$ (features do not have to include $S^*_1$ — see master document, Section 4.4)
  * 3D: $S^*_3$ → $I = \dots$
  * 4D: $S^*_4$ → $I = \dots$
* **Explicit UI warning (not decorative, but from Section 4.5 of the master document):** there is no statement of statistical significance next to the $I$ value — only the number. The UI must not imply "proof" of the result.
* **Interaction:** clicking on a branch — instant visualizer rebuild on the new set of axes (not a collapse animation — see next section).
* **Optional "NMI Threshold" filter:** a toggle next to the branch list, **off by default** — until enabled, the branch list does not change. When enabled, a slider appears (0–99%); the list only keeps branches whose $\mathrm{NMI} > $ the selected threshold — this is purely client-side filtering of already computed branches (no recalculation). A branch that passes the filter is initially rendered in its own maximum dimensionality (its $d$), just like without a filter; within it, all 4D↔3D↔2D↔1D collapse/split animations remain available (Section 4 below) — the filter only affects which branches are visible and selectable in the list, not the behavior of an already selected branch. If the currently selected branch stops passing the threshold, the branch with the highest dimensionality among those passing the filter is automatically selected; if no branch passes, the visualizer list is cleared with an explicit message (not to be confused with "no branches found" — the dataset has nothing to do with it, it's the user's threshold choice).

## 4. Visualizer (Discrete Center Workspace) Section

The workspace rendering the selected branch as a grid of discrete centers (master document, Section 5): a circle at the intersection of categories, size = number of objects, color = 4-zone purity scale (green / yellow / brown / red).

* **Within one branch**, the user can collapse/split the dimensionality (e.g., hide the 3rd axis for a selected 3D branch and see the 2D aggregate of the same features) — this is an animated transition maintaining the relative sizes of the circles (Object Constancy, master document Section 5.6, case 2).
* **Switching to another branch** (another independently found dimensionality) — an instant rebuild without a collapse animation, because the set of features might completely change (master document Section 5.6, case 1). The UI must visually distinguish these two actions so as not to create a false impression that branches are nested within each other.
* **4D:** third axis + slice stepper under the chart (film strip, master document Section 5.5), with play/pause/speed on top of the discrete slices.

---

## 5. User Flow

1. The user loads a dataset and selects a target column (e.g., "Edibility") and a criterion (e.g., "Poisonous").
2. The system computes up to 4 branches and shows their list: 1D (`Odor` → $I=0.90$), 2D, 3D (`Odor + Spore print color + Ring type` → $I=0.99$), 4D — without significance statements.
3. The user clicks on the 3D branch. The visualizer instantly rebuilds: a cube of discrete centers, "Poisonous" — part of the purity scale tending towards red.
4. The user collapses the 3D branch to 2D right in the visualizer — sees a circle-merging animation that preserves their relative size.
5. The user returns to the branch list and clicks on the independently found 2D branch (which might consist of another two features, not the first two of the 3D branch) — the scene rebuilds instantly, without animation.
