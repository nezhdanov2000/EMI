# Visual Sufficiency Framework (VSF)

VSF is an information-theoretic framework for adaptive visualization dimensionality selection. It analyzes a dataset and automatically determines the minimal necessary and sufficient number of visual channels (from 2D to 7D) required to faithfully represent the data structure.

## Project Structure

*   `vsf/` — Mathematical core of the VSF library. Contains discretization algorithms, mutual information calculations, FDR control, and dimensionality routing.
*   `server.py` — Local web server and REST API serving the interactive visualization.
*   `index.html` — Interactive visualization web interface (3D Scatter Plot powered by Plotly.js).
*   `graph.html` — NMI reasoning graph and logical inference interface.
*   `static/` — Static assets (CSS, JS).
*   `data/` — Dataset directory (e.g., `mushrooms.csv`).

## Quick Start

### Requirements
* Python 3.8+
* Dependencies (from `setup.py`): `numpy`, `scipy`, `pandas`.

You can install dependencies using `pip`:
```bash
pip install -e .
```

### Starting the Server
Start the local server using Python:
```bash
python server.py
```

Once the server is running, open your browser and navigate to: [http://localhost:8050](http://localhost:8050)

To open the Reasoning Graph interface, go to: [http://localhost:8050/graph.html](http://localhost:8050/graph.html)

## Features
*   **Perceptually-Matched Discretization (PMD):** Optimal quantization of continuous features constrained by visual channel capacities based on Rate-Distortion Theory.
*   **Adaptive Visual Routing (AVR):** Automatic selection of 1 to 7 axes based on Conditional Permutation Testing, FDR Control (Benjamini-Hochberg), and Normalized Mutual Information (NMI).
*   **Explainable AI (XAI):** Built-in textual warnings when data complexity exceeds 7D visual bandwidth, or when selected candidate features are statistically indistinguishable from noise.
*   **Graph Reasoning & Inference:** Automatic mining of statistically sound mediator chains and logical inference paths across feature hierarchies.

For detailed theoretical derivations and algorithmic specifications, see `Project_Master_Document.md`.
