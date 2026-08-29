# Visual Sufficiency Framework (VSF)

VSF is an information-theoretic system for the adaptive selection of visualization dimensionality. It analyzes a dataset and automatically determines the minimum necessary and sufficient number of visual channels (from 2D to 7D) to display the data structure.

## Project Structure

*   `vsf/` — Installable VSF library (`pip install -e .`). Mathematical core: discretization algorithms, mutual information calculation, dimensionality selection, mining, and visualization. Does not contain specific dataset dictionaries — see `translations` below. Includes `vsf/dashboard.py` (`export_full_dashboard` — standalone offline export of the entire interactive web interface into a single HTML file, see section below) and assets bundled with the library `vsf/templates/` (CSS/JS/HTML templates that `export_full_dashboard` reads via `importlib.resources`).
*   `examples/mushroom_demo.py` — Demo dataset-specific (UCI Mushroom) human-readable column/value names (`MUSHROOM_TRANSLATIONS`). Not part of the `vsf` library; this is what the *calling code* must provide for its dataset.
*   `server.py` — Local web server and REST API that serves the demo dataset visualization; imports `MUSHROOM_TRANSLATIONS` from `examples/` and passes it to `vsf` as `translations`.
*   `index.html` — Visualizer web interface (3D Scatter Plot using Plotly.js).
*   `static/` — Static resources (CSS, JS).
*   `data/` — Directory with datasets (e.g., `mushrooms.csv`).
*   `tests/` — Regression test suite for pytest (not included in the installable package).
*   `pyproject.toml` — Single source of truth for package metadata (replaced `setup.py`).

## Quick Start

### Requirements
* Python **3.10+** (not 3.8 — some modules use the `X | None` syntax
  (PEP 604) in annotations without `from __future__ import annotations`, which
  crashes with a `TypeError` on import on 3.8/3.9).
* Dependencies are declared in `pyproject.toml`: `numpy`, `pandas`.
  Optional extras: `vsf[gpu]` (GPU backend via `cupy` —
  `vsf/backend.py` uses it as an optional, switchable backend with a
  fallback to `numpy`) and `vsf[test]` (`pytest`, for running `tests/`).

Install the package in editable mode (for reproducibility of the paper's
results; publication to PyPI is not planned):
```bash
pip install -e .
# or, together with test dependencies:
pip install -e ".[test]"
```

### Running the server
Run the local server using Python (from the repository root,
after `pip install -e .`):
```bash
python server.py
```

After starting the server, open your browser and navigate to: [http://localhost:8050](http://localhost:8050)

### Using `vsf` with your own dataset
The library is not tied to the mushroom dataset — it accepts an optional
`translations` argument (see `vsf.vis.Translations`) for human-readable
labels; without it, raw values from the data are used:
```python
import vsf

# Without translations: labels = raw values/column names.
payload = vsf.prepare_visualization_payload(res, X, Z, feature_names=feature_names)

# With translations (following examples/mushroom_demo.py) — your own dictionary
# for your dataset, not modifying the library:
my_translations = {"columns": {...}, "values": {...}}
payload = vsf.prepare_visualization_payload(
    res, X, Z, feature_names=feature_names, translations=my_translations
)
```

### Standalone HTML Dashboard (`vsf.export_full_dashboard`)
The entire interactive web interface (`index.html` + `graph.html` +
`static/` + `server.py`) can be packed into a **single self-contained
HTML file**, which can be opened locally (`file:///...`) without running
`server.py` — no CORS restrictions and network requests to the backend,
only Plotly.js and Google Fonts from a CDN:
```python
import pandas as pd
import vsf

df = pd.read_csv("data/mushrooms.csv")
html = vsf.export_full_dashboard(df, target="class", criterion="p")
open("dashboard.html", "w", encoding="utf-8").write(html)
```
The resulting file includes: interactive switching of X/Y/Z axes and
1D/2D/3D/4D modes on embedded data, a Rule Explorer panel
("dirty" discrete centers + conjunctive rule mining), XAI HUD
(A/B/C/D scenario, `d*`, VIR, explanation), and a fully offline
Graph Inference (logical inference chains and super-links catalog,
ported from `graph.html`).

**Important, deliberate limitations** (unlike the live `server.py`):
*   **One** `target`/`criterion` scenario is "baked" into a single export —
    other catalog leaves are shown for reference but are inactive. If you need
    another scenario — call `export_full_dashboard` again with different
    `target`/`criterion`.
*   The composite AND-filter ("Filter Search") is not ported — it
    restarts the full permutation-tested AVR feature selection on an
    arbitrary user condition, which cannot be exhaustively
    precomputed in a static file. Use the live `server.py` for this.

All statistics (feature selection, purity, Miller-Madow adjusted
NMI, Benjamini-Hochberg significance) are computed **in Python**
when calling `export_full_dashboard` — the embedded JavaScript never
recalculates p-values or retrains the model, it only filters/counts
already validated data (see docstring `vsf/dashboard.py`).

## Features
*   **Perceptually-Aligned Discretization (PMD):** Optimal quantization of continuous features taking into account the bandwidth of visual channels.
*   **Adaptive Visual Routing (AVR):** Automatic selection of 1 to 7 axes based on the Permutation Test and Mutual Information metric.
*   **Explainable AI (XAI):** The system outputs text warnings if the data structure is too complex to be visualized in 7D, or if the selected features are noise.

For a detailed scientific description of the algorithms, please refer to the `Project_Master_Document.md` file.

## License

MIT — see the `LICENSE` file.
