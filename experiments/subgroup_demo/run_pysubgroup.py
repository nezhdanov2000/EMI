"""
Subgroup discovery (pysubgroup) на том же titanic.csv, что использует VSF.

Запуск:
    python run_pysubgroup.py                       # по умолчанию: survived, глубина 2, порог чистоты 0.8
    python run_pysubgroup.py --target died --depth 3 --purity 0.9

Что делает:
  1. Перебирает все правила вида "col1=a AND col2=b AND ..." до заданной глубины.
  2. Для каждого считает size (сколько строк), purity (доля целевого класса),
     coverage (доля целевого класса, попавшая в правило).
  3. Показывает топ правил по purity среди тех, что прошли порог и минимальный размер.
  4. Показывает, какое покрытие даёт объединение всех прошедших правил.
Никакой защиты от переобучения здесь нет: чистота измерена на тех же строках,
на которых правило найдено. Это как раз то, о чём надо помнить при сравнении с VSF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysubgroup as ps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parents[2] / "benchmark_data" / "titanic.csv"))
    ap.add_argument("--target-col", default="survived")
    ap.add_argument("--target", default="survived", help="значение целевого столбца")
    ap.add_argument("--depth", type=int, default=2, help="макс. число условий в правиле (1D..4D)")
    ap.add_argument("--purity", type=float, default=0.8, help="порог чистоты")
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = (df[args.target_col] == args.target).to_numpy()
    n_pos = int(y.sum())
    print(f"Строк: {len(df)}, целевых ({args.target_col}={args.target}): {n_pos}, база {n_pos/len(df):.3f}")

    target = ps.BinaryTarget(args.target_col, args.target)
    search_space = ps.create_selectors(df, ignore=[args.target_col])
    # StandardQF(a=1) = size * (p - p0): классическая "интересность" subgroup discovery
    task = ps.SubgroupDiscoveryTask(
        df, target, search_space,
        result_set_size=5000, depth=args.depth,
        qf=ps.StandardQF(a=1.0),
        constraints=[ps.MinSupportConstraint(args.min_size)],
    )
    result = ps.Apriori().execute(task)

    rows = []
    for _score, sg in result.to_descriptions():
        mask = sg.covers(df)
        size = int(mask.sum())
        hits = int(y[mask].sum())
        rows.append({
            "rule": str(sg),
            "n_conds": len(sg.selectors),
            "size": size,
            "purity": hits / size,
            "coverage": hits / n_pos,
        })
    res = pd.DataFrame(rows)
    ok = res[res.purity >= args.purity].sort_values(["purity", "size"], ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nПравил найдено: {len(res)}; с purity >= {args.purity} и size >= {args.min_size}: {len(ok)}\n")
    print(ok.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    union = np.zeros(len(df), dtype=bool)
    for rule in ok.rule:
        for _s, sg in result.to_descriptions():
            if str(sg) == rule:
                union |= sg.covers(df)
                break
    print(f"\nОбъединение всех прошедших правил: покрытие целевого класса "
          f"{y[union].sum()/n_pos:.3f}, чистота объединения {y[union].mean() if union.any() else 0:.3f}")


if __name__ == "__main__":
    main()
