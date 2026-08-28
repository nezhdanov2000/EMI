"""
One-off script: a CURATED multi-scenario export — a smaller demonstration
of the multi-target scenario precomputation feature (see vsf/dashboard.py's
`targets` parameter) that stays comfortably under both the chat delivery
limit (30 MB) and the device-folder sync limit (20 MB).

The full default (`targets=None`, all 23 mushroom columns) produces a
~69 MB file — functionally correct (verified by tests/test_dashboard.py
and a Playwright browser smoke test switching between class/odor/habitat/
back-to-class), but too large to transfer through chat or the device
bridge. This script picks 5 additional, well-separated columns instead of
all 22, for a 6-scenario file (primary "class" + 5) sized for delivery.

For the COMPLETE all-columns export, run export_dashboard.py directly in
this environment (C:\\xampp\\htdocs\\7D) instead — it writes straight to
local disk, so no transfer-size limit applies there.
"""
import os

import pandas as pd

import vsf
from examples.mushroom_demo import MUSHROOM_TRANSLATIONS

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "mushrooms.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "vsf_full_dashboard_demo.html")

df = pd.read_csv(DATASET_PATH)

html = vsf.export_full_dashboard(
    df,
    target="class",
    criterion="p",
    targets=["odor", "habitat", "population", "cap-shape", "bruises"],
    translations=MUSHROOM_TRANSLATIONS,
    alpha=0.01,
    vir_threshold=0.85,
    n_permutations=100,
    random_state=42,
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_PATH} ({len(html):,} bytes)")
