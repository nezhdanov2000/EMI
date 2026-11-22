import os
import sys
import json
from pathlib import Path

cache = Path("experiments/results/cache")
cache_extra = Path("experiments/results/cache_extra")

mapping = {
    "connect_4_db4cb0600ba7c24c": ("connect_4 loss purity", 5),
    "connect_4_f87525c9ad1190b6": ("connect_4 loss certified", 5),
    "connect_4_4d355ffeafcd9cc6": ("connect_4 loss split", 5),
    "connect_4_4fa00a611aba92df": ("connect_4 win purity", 5),
    "connect_4_f5a255dd167799a2": ("connect_4 win certified", 5),
    "connect_4_b7be9262903d8c88": ("connect_4 win split", 5),
    "chess_krkp_ed5ff4ad2beb06ae": ("chess_krkp won purity", 15),
    "chess_krkp_967b5dc13425d44a": ("chess_krkp won certified", 15),
    "chess_krkp_73e3937f46476491": ("chess_krkp won split", 15),
}

print("=== PROGRESS SUMMARY ===")
for prefix, (desc, total) in mapping.items():
    c_splits = set()
    for f in cache.glob(f"{prefix}_split*.json"):
        c_splits.add(int(f.stem.split("_split")[1]))
    
    extra_methods = {}
    for f in cache_extra.glob(f"{prefix}_split*.json"):
        s = int(f.stem.split("_split")[1])
        try:
            with open(f) as fp:
                d = json.load(fp)
                for m in d.get("methods", {}):
                    extra_methods.setdefault(m, set()).add(s)
        except Exception:
            pass
    m_info = ", ".join(f"{m}:{len(splits)}/{total}" for m, splits in sorted(extra_methods.items()))
    print(f"{desc:26}: main={len(c_splits):2d}/{total} (missing {sorted(list(set(range(total))-c_splits))}) | extra: {m_info}")

import datetime

all_files = sorted(list(cache.glob('*')) + list(cache_extra.glob('*')), key=lambda p: p.stat().st_mtime, reverse=True)
print("\n=== RECENTLY MODIFIED CACHE FILES ===")
for p in all_files[:15]:
    t = datetime.datetime.fromtimestamp(p.stat().st_mtime)
    print(f"{t.strftime('%Y-%m-%d %H:%M:%S')} | {p.parent.name:11} | {p.name}")

