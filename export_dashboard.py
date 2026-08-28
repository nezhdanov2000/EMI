"""
One-off script: re-export the standalone VSF dashboard after the Phase 2
architecture cleanup (4D time/frame controller, hard 4D cap, graph removal).

Mirrors server.py's live-app AVR parameters (alpha=0.01, vir_threshold=0.85,
n_permutations=100, random_state=42) for target="class", criterion="p"
(poisonous) on the UCI Mushroom dataset, so the exported scenario matches
what the live app would compute for the same inputs. `max_d` is left at its
new default (`vsf.dashboard._MAX_SUPPORTED_D` == 4) rather than passed
explicitly, since export_full_dashboard now rejects anything higher anyway.
"""
import os

import pandas as pd

import vsf
from examples.mushroom_demo import MUSHROOM_TRANSLATIONS

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "mushrooms.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "vsf_full_dashboard.html")

df = pd.read_csv(DATASET_PATH)

html = vsf.export_full_dashboard(
    df,
    target="class",
    criterion="p",
    translations=MUSHROOM_TRANSLATIONS,
    alpha=0.01,
    vir_threshold=0.85,
    n_permutations=100,
    random_state=42,
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_PATH} ({len(html):,} bytes)")
