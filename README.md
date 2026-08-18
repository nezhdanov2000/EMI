# Visual Sufficiency Framework (VSF)

VSF is an information-theoretic framework for adaptive visualization dimensionality selection. It analyzes datasets and mathematically determines the minimum necessary and sufficient visual encoding channels (from 2D up to 7D) required to faithfully represent data structure.

## Project Structure

*   `vsf/` — Mathematical core library of VSF. Implements discretization, mutual information computation, and adaptive dimensionality selection algorithms.
*   `server.py` — Local web server and REST API delivering dynamic dataset analysis and visualization payloads.
*   `index.html` — Interactive WebGL 3D visualizer interface powered by Plotly.js and glassmorphism UI.
*   `static/` — Frontend assets (CSS stylesheets, JS controller).
*   `data/` — Datasets directory (e.g. `mushrooms.csv`).
*   `tests/` — Test suites for mathematical and routing components.

## Quick Start

### Requirements
* Python 3.8+
* Dependencies (from `setup.py`): `numpy`, `scipy`, `pandas`.

Install dependencies with `pip`:
```bash
pip install -e .
```

### Starting the Web Server
Launch the local web server:
```bash
python server.py
```

Then open your browser and navigate to: [http://localhost:8050](http://localhost:8050)

## Core Features
*   **Perceptually-Matched Discretization (PMD):** Optimal quantization of continuous features constrained by visual channel capacities (Rate-Distortion Theory).
*   **Adaptive Visual Routing (AVR):** Automatic selection of 1 to 7 visual axes using Permutation Tests and Mutual Information (MI).
*   **Explainable AI (XAI):** Contextual diagnostic insights explaining data predictability, noise filtration, and high-dimensional projection warnings.

For full mathematical derivations and formal proofs, refer to `Project_Master_Document.md`.
