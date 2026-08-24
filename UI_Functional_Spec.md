# VSF UI & Functional Specification

This document describes the user interface (UI) architecture and interaction logic for the Visual Sufficiency Framework (VSF).

## 1. Interface Concept (Master-Detail View)

The interface follows the Master-Detail architectural pattern and is logically divided into two primary zones:
1. **Insights Catalog (Section 2 - Master):** Navigation and exploratory panel for inspecting global dataset dependencies discovered during the Auto-Discovery process.
2. **Adaptive Visualizer (Section 1 - Detail):** Interactive workspace where the specific multidimensional visual configuration (from 1D to 7D) selected by the user in the Catalog is dynamically rendered.

---

## 2. Section 2: Insights Catalog (Global System Ranking)

This section is structured as a hierarchical accordion / tree view reflecting the propositionalization (One-vs-Rest) pipeline.

### Level 1: Characteristics (Features)
* **Description:** List of all dataset columns (e.g., Edibility, Odor, Cap Color).
* **Metrics:** For each characteristic:
  * `Max MI`: Maximum predictive power (Mutual Information) discovered for this feature.
  * `Optimal d*`: Dimensionality of the minimal sufficient basis achieving this Max MI.
* **Interaction:** Clicking a Characteristic expands its list of Criteria (Atomic Predicates).

### Level 2: Criteria (Atomic Predicates / Outcomes)
* **Description:** Binarized outcomes of a specific characteristic (e.g., for Odor: "Almond", "Anise", "Foul").
* **Metrics:** 
  * `Max MI` for the specific criterion.
  * `Optimal d*` (optimal number of visual axes required to explain this outcome).
* **Interaction:** Clicking a Criterion expands its Top Systems (Pareto frontier of feature subsets).

### Level 3: Top Systems (Pareto Frontier of Feature Subsets)
* **Description:** Ordered list of best-performing feature subsets (bases $S^*$), sorted by increasing MI and minimal dimensionality (from 1D up to $d^*$). Only Pareto-optimal solutions are presented:
  * Best 1D system (e.g., `[Stalk Shape]`) $\to MI = 0.45$
  * Best 2D system (`[Stalk Shape, Spore Print Color]`) $\to MI = 0.82$
  * Best 3D system (`[Stalk Shape, Spore Print Color, Population]`) $\to MI = 0.96$
* **Interaction:** Clicking a specific system sends an event to Section 1 to instantly render this coordinate basis.

---

## 3. Section 1: Adaptive Visualizer (AVR Workspace)

The main workspace occupying the central area of the display.

* **Operational Logic:** Receives state updates from the Insights Catalog (target predicate $Z$ and selected basis $S^*$).
* **Adaptive Dimensionality:** Depending on system dimensionality (from 1D to 7D), the visualizer automatically selects the appropriate mode:
  * **1D-3D:** Minimalist spatial coordinate plot (2D/3D Scatter plot / Bubble chart).
  * **4D-7D:** Spatial coordinates augmented with color encodings (Hue, Lightness) and time (animation) in strict accordance with the AVR algorithm.
* **XAI Feedback:**
  * If dimensionality $d^* < 7$ and the system fully captures the criterion (Scenarios A & B) — an explanatory confirmation pill with $MI$ metrics is displayed.
  * If dimensionality reaches the 7D limit but residual structure remains uncaptured (Scenario C) — a warning is displayed: *"Rendered best 7D projection. Information loss: X%"*.

---

## 4. User Flow Scenario

1. The user loads a dataset. The backend executes the Auto-Discovery pipeline.
2. In **Section 2**, the ranking appears: characteristic "Edibility" exhibits `Max MI = 0.99`.
3. The user clicks "Edibility" and views the criterion "Poisonous".
4. Clicking "Poisonous" expands the Top Systems. The user observes that a 1D system (Odor alone) achieves `MI = 0.90`, while a 3D system (Odor + Spore Print Color + Ring Type) achieves `MI = 0.99`.
5. The user selects the 3D system.
6. In **Section 1**, the adaptive 3D scatter plot is rendered immediately: poisonous mushrooms are highlighted in red and cleanly segregated into dense clusters along the three selected axes.
