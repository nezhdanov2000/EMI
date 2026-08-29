"""
One-off script: re-export the standalone VSF dashboard for the UCI Mushroom
demo dataset under the v2.0 "Clean Core" architecture (Independent Branch
Discovery, up to 4 independently-found branches per export — see
Project_Master_Document.md Section 4).

target="class", criterion="p" (poisonous). `discover_branches` is fully
deterministic (Project_Master_Document.md Section 4.5) — there is no
alpha/vir_threshold/n_permutations/random_state to pin anymore, unlike the
v1.0 AVR engine this replaced. `max_d` is left at its default
(`vsf.avr.MAX_BRANCH_D` == 4).
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
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {OUTPUT_PATH} ({len(html):,} bytes)")
