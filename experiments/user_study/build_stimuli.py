"""
Stimuli for the readability study: the same result shown three ways.

For one dataset, target value, purity floor and size floor (observed-purity
rule, all rows) this script computes

* the VSF partial-centre branch of one dimensionality, with the call the
  interactive app makes (`vsf.partial.discover_partial_branches`), so the
  answer key matches what the app draws;
* a list of free disjoint rules at the SAME condition budget the VSF branch
  spends (`baselines.select_rules(disjoint=True)`);
* one CART tree chosen as in the paper (`baselines.select_tree`) at that
  budget, refitted here to draw it;

and writes, under `experiments/user_study/out/<dataset>/`:

* `rule_list.html` - the competitor screen for the rule list (sortable table);
* `tree.html`      - the competitor screen for the decision tree (SVG);
* `heatmap.html`   - the cells of the VSF schema as flat tables (the same
  content as the lattice, in two dimensions);
* `rule_list_same_groups_as_grid.html` - the VSF groups themselves written
  as a list: same content as the grid, different display;
* `answer_key.md`  - the groups of every system, three case cards and the
  correct answer of every task of the question paper, per system.

The two HTML screens carry no method name: the moderator assigns the letters
A / B / C. Deterministic (seed 0); no network.

    python experiments/user_study/build_stimuli.py --dataset attrition --positive Yes --tau 0.5 --m 20 --d 3
"""
from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from baselines import select_rules, select_tree  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vsf.centers import CenterSpec  # noqa: E402
from vsf.partial import discover_partial_branches  # noqa: E402

HIGHLIGHT = "#d55e00"
SEED = 0


@dataclass(frozen=True)
class StudyGroup:
    """One highlighted group of one system: its conditions, members and counts."""

    conditions: Tuple[Tuple[str, str, bool], ...]  # (column, value, equals?)
    members: np.ndarray  # bool over all rows
    n: int
    k: int

    @property
    def share(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def columns(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for col, _, _ in self.conditions:
            if col not in seen:
                seen.append(col)
        return tuple(seen)

    def text(self) -> str:
        return " AND ".join(f"{c} {'=' if eq else '≠'} {v}" for c, v, eq in self.conditions)


@dataclass(frozen=True)
class TreeNode:
    """A node of the drawn tree. Leaves have `column is None`."""

    node_id: int
    depth: int
    n: int
    k: int
    column: Optional[str]
    value: Optional[str]
    no_child: Optional[int]
    yes_child: Optional[int]


# --------------------------------------------------------------------------
# The three systems
# --------------------------------------------------------------------------
def encode(X: np.ndarray) -> Tuple[np.ndarray, List[List[str]]]:
    """Integer codes per column (levels in sorted order), as `experiments/baselines.py` expects, and the levels."""
    codes = np.empty(X.shape, dtype=np.int64)
    levels: List[List[str]] = []
    for j in range(X.shape[1]):
        uniq, inv = np.unique(X[:, j].astype(str), return_inverse=True)
        codes[:, j] = inv
        levels.append([str(u) for u in uniq])
    return codes, levels


def vsf_groups(X: np.ndarray, z: np.ndarray, feats: Sequence[str], tau: float, m: int, d: int) -> Tuple[Tuple[str, ...], List[StudyGroup]]:
    """The app's partial-centre branch of dimensionality `d` under the observed-purity rule."""
    spec = CenterSpec(tau=tau, alpha=0.05, rule="purity", min_samples=m, multiplicity="bonferroni")
    _, partial = discover_partial_branches(
        X, z, feature_names=list(feats), center_spec=spec, positive_class=1, max_d=4,
        direction="presence", prune_dependent=False, features=None, family_max_d=4, random_state=SEED,
    )
    if d not in partial:
        raise SystemExit(f"no partial branch of dimensionality {d}; available: {sorted(partial)}")
    branch = partial[d]
    groups: List[StudyGroup] = []
    for centre in branch.own.centres:
        row = int(np.flatnonzero(centre.rows)[0])
        conds = tuple((feats[j], str(X[row, j]), True) for j in centre.sub_schema)
        groups.append(StudyGroup(conds, np.asarray(centre.rows, dtype=bool), int(centre.n), int(centre.k)))
    return tuple(feats[j] for j in branch.schema), groups


def exploration_answers(X: np.ndarray, z: np.ndarray, feats: Sequence[str], tau: float, m: int, d: int, tau2: float) -> List[Tuple[str, str]]:
    """
    Answers of the tasks only the interactive app can answer: the coverage of
    the branch of dimensionality `d` after the boundary is moved to `tau2`,
    the branch with the highest coverage at the original boundary, and the
    interval of the first listed centre of the `d` branch.
    """
    def branches(floor: float) -> Dict[int, object]:
        spec = CenterSpec(tau=floor, alpha=0.05, rule="purity", min_samples=m, multiplicity="bonferroni")
        _, partial = discover_partial_branches(
            X, z, feature_names=list(feats), center_spec=spec, positive_class=1, max_d=4,
            direction="presence", prune_dependent=False, features=None, family_max_d=4, random_state=SEED,
        )
        return partial

    base, moved = branches(tau), branches(tau2)
    out: List[Tuple[str, str]] = []
    if d in moved:
        v = moved[d].own
        out.append((f"12. Coverage on the {d}D card after the boundary is moved to {100 * tau2:.0f}%",
                    f"{100 * v.selection.coverage:.1f}% ({len(v.centres)} centres; schema {' + '.join(feats[j] for j in moved[d].schema)})"))
    else:
        out.append((f"12. Coverage on the {d}D card after the boundary is moved to {100 * tau2:.0f}%", "no centres (0%)"))
    best = max(base, key=lambda k: (base[k].own.selection.coverage, -k))
    vb = base[best].own
    out.append(("13. Card with the highest coverage at the original boundary",
                f"{best}D: {100 * vb.selection.coverage:.1f}%, {len(vb.centres)} centres, {vb.selection.conditions} conditions"))
    first = base[d].own.centres[0]
    sure = first.purity_lower >= tau
    out.append((f"14. Interval of centre #1 on the {d}D card",
                f"{100 * first.purity:.1f}% [{100 * first.purity_lower:.1f}, {100 * first.purity_upper:.1f}]; "
                f"lower bound {'at or above' if sure else 'below'} the boundary: {'Yes' if sure else 'No'}"))
    return out


def rule_groups(X: np.ndarray, z: np.ndarray, feats: Sequence[str], tau: float, m: int, budget: int) -> List[StudyGroup]:
    """Free disjoint rules of at most four conditions, greedy under the given condition budget."""
    train = np.arange(len(z))
    codes, levels = encode(X)
    # the baseline labels a rule "<name>=<integer code>"; column indices as names keep the parse unambiguous
    result = select_rules(codes, train, z, tau, m, [budget], [str(j) for j in range(len(feats))], disjoint=True)
    groups: List[StudyGroup] = []
    for g in result.groups[budget]:
        pairs = [part.split("=", 1) for part in g.label.split(" & ")]
        conds = tuple((feats[int(j)], levels[int(j)][int(v)], True) for j, v in pairs)
        members = np.asarray(g.members, dtype=bool)
        groups.append(StudyGroup(conds, members, int(members.sum()), int(z[members].sum())))
    return groups


def tree_groups(X: np.ndarray, z: np.ndarray, feats: Sequence[str], tau: float, m: int, budget: int) -> Tuple[Dict[int, TreeNode], List[StudyGroup], List[StudyGroup]]:
    """
    The CART tree `baselines.select_tree` picks at `budget`, refitted to
    read its structure. Highlighted groups are ALL leaves with at least `m`
    rows and target share >= tau (what a reader of the diagram sees), which
    is the set `select_tree` chose from before applying the budget.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    train = np.arange(len(z))
    codes, levels = encode(X)
    picked = select_tree(codes, train, z, tau, m, [budget], random_state=SEED)
    chosen = picked.groups[budget]
    if not chosen:
        raise SystemExit("the tree baseline selects no qualifying leaf at this budget")
    criterion, weight, depth_token = chosen[0].label.rsplit("-leaf", 1)[0].split("-")
    class_weight = None if weight == "None" else weight
    depth = int(depth_token.replace("depth", ""))
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    H = encoder.fit_transform(codes)
    tree = DecisionTreeClassifier(criterion=criterion, class_weight=class_weight, max_depth=depth,
                                  min_samples_leaf=max(1, m), random_state=SEED).fit(H, z)
    one_hot: List[Tuple[str, str]] = [(feats[j], levels[j][int(v)]) for j, cats in enumerate(encoder.categories_) for v in cats]
    t = tree.tree_
    leaf_of = tree.apply(H)
    path = tree.decision_path(H)
    nodes: Dict[int, TreeNode] = {}

    def walk(node: int, depth_now: int) -> None:
        rows = np.asarray(path[:, node].todense()).ravel().astype(bool)
        n, k = int(rows.sum()), int(z[rows].sum())
        if t.children_left[node] == -1:
            nodes[node] = TreeNode(node, depth_now, n, k, None, None, None, None)
            return
        col, val = one_hot[int(t.feature[node])]
        # one-hot split "x <= 0.5": left = the value is absent (no), right = present (yes)
        nodes[node] = TreeNode(node, depth_now, n, k, col, val, int(t.children_left[node]), int(t.children_right[node]))
        walk(int(t.children_left[node]), depth_now + 1)
        walk(int(t.children_right[node]), depth_now + 1)

    walk(0, 0)
    conditions: Dict[int, Tuple[Tuple[str, str, bool], ...]] = {0: ()}
    for node_id in sorted(nodes, key=lambda i: nodes[i].depth):
        node = nodes[node_id]
        if node.column is None:
            continue
        assert node.value is not None and node.no_child is not None and node.yes_child is not None
        conditions[node.no_child] = conditions[node_id] + ((node.column, node.value, False),)
        conditions[node.yes_child] = conditions[node_id] + ((node.column, node.value, True),)
    groups: List[StudyGroup] = []
    leaves: List[StudyGroup] = []
    for node_id, node in nodes.items():
        if node.column is not None:
            continue
        leaf = StudyGroup(conditions[node_id], leaf_of == node_id, node.n, node.k)
        leaves.append(leaf)
        if node.n >= m and node.k / node.n >= tau:
            groups.append(leaf)
    groups.sort(key=lambda g: -g.k)
    return nodes, groups, leaves


def grid_cells(X: np.ndarray, z: np.ndarray, feats: Sequence[str], schema: Sequence[str]) -> List[StudyGroup]:
    """Every occupied cell of the grid of `schema` (no level merging: the caller checks the capacity rule is idle)."""
    cols = [feats.index(c) for c in schema]
    keys = np.array(["\x1f".join(str(v) for v in row) for row in X[:, cols]])
    cells: List[StudyGroup] = []
    for key in np.unique(keys):
        members = keys == key
        values = key.split("\x1f")
        cells.append(StudyGroup(tuple((c, v, True) for c, v in zip(schema, values)), members,
                                int(members.sum()), int(z[members].sum())))
    return cells


def heatmap_html(cells: Sequence[StudyGroup], groups: Sequence[StudyGroup], schema: Sequence[str], header: str,
                 target: str, positive: str) -> str:
    """
    The cells of the VSF schema as flat tables: the same content as the
    lattice, without perspective. Rows are the axis with the most levels,
    columns the next, and one table per level of the third axis. A cell
    prints its rows and its target share, is shaded by the share, and
    carries the number of the highlighted group it belongs to (thick border).
    """
    if not 2 <= len(schema) <= 3:
        raise SystemExit("the heat map is drawn for schemas of two or three columns")
    levels = {c: sorted({dict((a, b) for a, b, _ in cell.conditions)[c] for cell in cells}) for c in schema}
    order = sorted(schema, key=lambda c: -len(levels[c]))
    row_axis, col_axis = order[0], order[1]
    panel_axis = order[2] if len(order) == 3 else None
    lookup = {tuple(dict((a, b) for a, b, _ in cell.conditions)[c] for c in order): cell for cell in cells}
    group_of: Dict[int, int] = {}
    for gi, g in enumerate(groups, start=1):
        for ci, cell in enumerate(cells):
            if bool(np.any(cell.members & g.members)):
                group_of[ci] = gi
    index_of = {id(cell): ci for ci, cell in enumerate(cells)}
    css = (_CSS + "table.hm td,table.hm th{border:1px solid #d9e2ec;text-align:center;padding:8px 10px;font-size:13px}"
           "table.hm th{cursor:default} .panel{display:inline-block;vertical-align:top;margin:0 26px 18px 0}"
           ".panel h2{font-size:15px;margin:0 0 6px} td.hit{outline:3px solid " + HIGHLIGHT + ";outline-offset:-3px}"
           " .g{font-weight:700;color:" + HIGHLIGHT + "} td.empty{color:#bcccdc}")
    panels: List[str] = []
    for pv in (levels[panel_axis] if panel_axis else [None]):
        head = "".join(f"<th>{html.escape(v)}</th>" for v in levels[col_axis])
        body: List[str] = []
        for rv in levels[row_axis]:
            tds: List[str] = []
            for cv in levels[col_axis]:
                cell = lookup.get((rv, cv, pv) if panel_axis else (rv, cv))
                if cell is None:
                    tds.append("<td class='empty'>no rows</td>")
                    continue
                gi = group_of.get(index_of[id(cell)])
                tag = f"<div class='g'>group {gi}</div>" if gi else ""
                tds.append(f"<td class='{'hit' if gi else ''}' style='background:rgba(47,111,183,{0.06 + 0.6 * cell.share:.2f})'>"
                           f"<b>{100 * cell.share:.0f}%</b><br>{cell.k} of {cell.n}{tag}</td>")
            body.append(f"<tr><th style='text-align:left'>{html.escape(rv)}</th>{''.join(tds)}</tr>")
        title = f"<h2>{html.escape(panel_axis)} = {html.escape(str(pv))}</h2>" if panel_axis else ""
        panels.append(f"<div class='panel'>{title}<table class='hm'><thead><tr><th style='text-align:left'>{html.escape(row_axis)} \\ "
                      f"{html.escape(col_axis)}</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    legend = (f"<p class='legend'>Each cell is one combination of values. It shows the share of its rows with {html.escape(target)} = "
              f"{html.escape(positive)} (darker = higher share) and the count, for example \"31 of 62\". Cells with a thick orange border "
              f"belong to a highlighted group; cells with the same group number form one group.</p>")
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Groups</title><style>{css}</style></head><body>"
            f"{header}{''.join(panels)}{legend}</body></html>")


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------
_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1f2933;margin:28px;background:#fff}
h1{font-size:20px;margin:0 0 6px} p.sub{margin:0 0 18px;color:#52606d;font-size:14px;max-width:900px}
table{border-collapse:collapse;font-size:15px} th,td{padding:9px 14px;border-bottom:1px solid #d9e2ec;text-align:left;vertical-align:middle}
th{cursor:pointer;user-select:none;background:#f0f4f8;font-weight:600;white-space:nowrap} th:hover{background:#d9e2ec}
td.num{text-align:right;font-variant-numeric:tabular-nums} .cond b{font-weight:600}
.and{color:#829ab1;font-size:12px;margin:0 6px} .bar{display:inline-block;height:12px;background:#2f6fb7;vertical-align:middle;margin-right:8px;border-radius:2px}
.legend{margin-top:16px;font-size:13px;color:#52606d}
"""


def _subtitle(title: str, target: str, positive: str, n: int, p: int, tau: float, m: int) -> str:
    return (f"<h1>{html.escape(title)}</h1><p class='sub'>{n:,} rows; in {p:,} of them ({100 * p / n:.0f}%) "
            f"<b>{html.escape(target)} = {html.escape(positive)}</b>. Highlighted groups have at least {m} rows and "
            f"at least {100 * tau:.0f}% of their rows with {html.escape(target)} = {html.escape(positive)}.</p>")


def rule_list_html(groups: Sequence[StudyGroup], header: str, target: str, positive: str) -> str:
    rows = []
    for i, g in enumerate(groups, start=1):
        cond = "<span class='and'>AND</span>".join(f"<b>{html.escape(c)}</b> = {html.escape(v)}" for c, v, _ in g.conditions)
        rows.append(f"<tr><td class='num'>{i}</td><td class='cond'>{cond}</td><td class='num' data-v='{g.n}'>{g.n:,}</td>"
                    f"<td class='num' data-v='{g.k}'>{g.k:,}</td><td data-v='{g.share:.6f}'>"
                    f"<span class='bar' style='width:{140 * g.share:.0f}px'></span>{100 * g.share:.0f}%</td></tr>")
    script = """
<script>
document.querySelectorAll('th').forEach(function(th, col){th.addEventListener('click', function(){
  var tb=document.querySelector('tbody'), rows=Array.from(tb.rows), dir=th.dataset.dir==='asc'?'desc':'asc'; th.dataset.dir=dir;
  rows.sort(function(a,b){var x=a.cells[col], y=b.cells[col];
    var u=x.dataset.v!==undefined?parseFloat(x.dataset.v):x.textContent.trim(), v=y.dataset.v!==undefined?parseFloat(y.dataset.v):y.textContent.trim();
    if(!isNaN(parseFloat(u))&&!isNaN(parseFloat(v))){u=parseFloat(u);v=parseFloat(v);} return (u>v?1:u<v?-1:0)*(dir==='asc'?1:-1);});
  rows.forEach(function(r){tb.appendChild(r);});});});
</script>"""
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Groups</title><style>{_CSS}</style></head><body>"
            f"{header}<table><thead><tr><th>#</th><th>Group (all conditions must hold)</th><th>Rows</th>"
            f"<th>Rows with {html.escape(target)} = {html.escape(positive)}</th><th>Share</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table><p class='legend'>Click a column heading to sort. No row of the data belongs to two groups.</p>"
            + script + "</body></html>")


def tree_html(nodes: Dict[int, TreeNode], tau: float, m: int, header: str, target: str, positive: str) -> str:
    box_w, box_h, gap_x, gap_y = 190, 62, 22, 70
    x_of: Dict[int, float] = {}
    next_leaf = [0]

    def place(node_id: int) -> float:
        node = nodes[node_id]
        if node.column is None:
            x = next_leaf[0] * (box_w + gap_x)
            next_leaf[0] += 1
        else:
            assert node.no_child is not None and node.yes_child is not None
            x = (place(node.no_child) + place(node.yes_child)) / 2
        x_of[node_id] = x
        return x

    place(0)
    width = next_leaf[0] * (box_w + gap_x) + 20
    height = (max(n.depth for n in nodes.values()) + 1) * (box_h + gap_y) + 20
    parts: List[str] = []
    for node_id, node in nodes.items():
        x, y = x_of[node_id] + 10, node.depth * (box_h + gap_y) + 10
        if node.column is not None:
            for child, word in ((node.no_child, "no"), (node.yes_child, "yes")):
                assert child is not None
                cx, cy = x_of[child] + 10 + box_w / 2, nodes[child].depth * (box_h + gap_y) + 10
                parts.append(f"<line x1='{x + box_w / 2:.0f}' y1='{y + box_h}' x2='{cx:.0f}' y2='{cy}' stroke='#829ab1' stroke-width='1.5'/>")
                parts.append(f"<text x='{(x + box_w / 2 + cx) / 2:.0f}' y='{(y + box_h + cy) / 2 + 4:.0f}' text-anchor='middle' "
                             f"font-size='13' fill='#334e68' stroke='#fff' stroke-width='4' paint-order='stroke'>{word}</text>")
    for node_id, node in nodes.items():
        x, y = x_of[node_id] + 10, node.depth * (box_h + gap_y) + 10
        share = node.k / node.n if node.n else 0.0
        hit = node.column is None and node.n >= m and share >= tau
        fill = f"rgba(47,111,183,{0.08 + 0.55 * share:.2f})"
        stroke, sw = (HIGHLIGHT, 4) if hit else ("#829ab1", 1.2)
        parts.append(f"<rect x='{x:.0f}' y='{y}' width='{box_w}' height='{box_h}' rx='6' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'/>")
        if node.column is not None:
            label = html.escape(f"{node.column} = {node.value}?")
            parts.append(f"<text x='{x + box_w / 2:.0f}' y='{y + 25}' text-anchor='middle' font-size='13' font-weight='600'>{label}</text>")
            parts.append(f"<text x='{x + box_w / 2:.0f}' y='{y + 46}' text-anchor='middle' font-size='12' fill='#334e68'>{node.n:,} rows · {100 * share:.0f}%</text>")
        else:
            parts.append(f"<text x='{x + box_w / 2:.0f}' y='{y + 26}' text-anchor='middle' font-size='14' font-weight='600'>{node.n:,} rows</text>")
            parts.append(f"<text x='{x + box_w / 2:.0f}' y='{y + 47}' text-anchor='middle' font-size='13'>{node.k:,} with {html.escape(positive)} · {100 * share:.0f}%</text>")
    legend = (f"<p class='legend'>Start at the top box and answer each question: follow <b>yes</b> or <b>no</b> down to a box at the bottom. "
              f"Each box shows its number of rows and the share with {html.escape(target)} = {html.escape(positive)} (darker = higher share). "
              f"Boxes with a thick orange border are the highlighted groups. Scroll sideways to see the whole tree.</p>")
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Groups</title><style>{_CSS}</style></head><body>"
            f"{header}<div style='overflow-x:auto'><svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' "
            f"font-family='system-ui,sans-serif'>{''.join(parts)}</svg></div>{legend}</body></html>")


# --------------------------------------------------------------------------
# Answer key
# --------------------------------------------------------------------------
def _columns(groups: Sequence[StudyGroup]) -> List[str]:
    out: List[str] = []
    for g in groups:
        for c in g.columns:
            if c not in out:
                out.append(c)
    return out


def pick_cases(systems: Dict[str, List[StudyGroup]], n_rows: int) -> List[int]:
    """
    Three real rows, chosen by a fixed rule (first row index that fits):
    one inside a highlighted group of every system, one inside none, and one
    on which the systems disagree (inside for some, outside for others).
    """
    inside = {name: np.any(np.vstack([g.members for g in groups]), axis=0) if groups else np.zeros(n_rows, dtype=bool)
              for name, groups in systems.items()}
    stack = np.vstack(list(inside.values()))
    wanted = (stack.all(axis=0), ~stack.any(axis=0), stack.any(axis=0) & ~stack.all(axis=0))
    cases: List[int] = []
    for mask in wanted:
        idx = np.flatnonzero(mask)
        if idx.size:
            cases.append(int(idx[0]))
    return cases


def context_answers(parts: Sequence[StudyGroup], groups: Sequence[StudyGroup], case_row: Optional[int], m: int) -> List[Tuple[str, str]]:
    """
    Answers of the context tasks for a system that shows a partition of all
    rows (`parts`: grid cells or tree leaves). A part is highlighted when it
    lies inside a highlighted group.
    """
    inside = np.any(np.vstack([g.members for g in groups]), axis=0)
    plain = [c for c in parts if not bool(np.any(c.members & inside))]
    out: List[Tuple[str, str]] = []
    if case_row is not None:
        part = next(c for c in parts if c.members[case_row])
        out.append(("9. Share in the group of Case 2", f"{100 * part.share:.0f}% ({part.k} of {part.n}; {part.text()})"))
    big = [c for c in plain if c.n >= m]
    best = max(big, key=lambda c: (c.share, c.n))
    ties = [c for c in big if abs(c.share - best.share) < 1e-12]
    out.append((f"10. Not highlighted, at least {m} rows, highest share",
                f"{best.text()}: {100 * best.share:.0f}% ({best.k} of {best.n})" + (f" [{len(ties)} tied]" if len(ties) > 1 else "")))
    largest = max(parts, key=lambda c: c.n)
    out.append(("11. Largest group shown, highlighted or not", f"{largest.text()}: {largest.n} rows"))
    return out


def answer_key(systems: Dict[str, List[StudyGroup]], X: np.ndarray, feats: Sequence[str], cases: Sequence[int],
               title: str, settings: str, vsf_schema: Sequence[str],
               partitions: Optional[Dict[str, List[StudyGroup]]] = None, m: int = 1,
               exploration: Optional[List[Tuple[str, str]]] = None) -> str:
    card_cols = [c for c in feats if any(c in _columns(g) for g in systems.values())]
    lines = [f"# Answer key: {title}", "", settings, "", "## Case cards (print on the system sheet)", ""]
    for i, row in enumerate(cases, start=1):
        lines.append(f"- **Case {i}** (row {row + 1} of the file): " + "; ".join(f"{c} = {X[row, feats.index(c)]}" for c in card_cols))
    for name, groups in systems.items():
        lines += ["", f"## {name}", ""]
        if name.startswith("VSF"):
            lines += [f"Schema on screen: {' x '.join(vsf_schema)}", ""]
        lines += ["| # | Group | Rows | With target | Share |", "| --- | --- | --- | --- | --- |"]
        for i, g in enumerate(groups, start=1):
            lines.append(f"| {i} | {g.text()} | {g.n} | {g.k} | {100 * g.share:.0f}% |")
        largest = max(groups, key=lambda g: g.n)
        purest = max(groups, key=lambda g: g.share)
        covered = int(np.any(np.vstack([g.members for g in groups]), axis=0).sum())
        overlap = covered != sum(g.n for g in groups)
        lines += ["", "| Task | Correct answer |", "| --- | --- |",
                  f"| 1. Number of highlighted groups | {len(groups)} |",
                  f"| 2. Columns used | {', '.join(_columns(groups))} |",
                  f"| 3. Largest group | {largest.text()} ({largest.n} rows) |",
                  f"| 4. Highest target share | {purest.text()} ({100 * purest.share:.0f}%) |"]
        for i, row in enumerate(cases, start=1):
            lines.append(f"| {4 + i}. Case {i} | {'Yes' if any(g.members[row] for g in groups) else 'No'} |")
        lines.append(f"| 8. Can one row be in two groups? | {'Yes' if overlap else 'No'} |")
        if partitions and name in partitions:
            case2 = cases[1] if len(cases) > 1 else None
            for task, answer in context_answers(partitions[name], groups, case2, m):
                lines.append(f"| {task} | {answer} |")
        if exploration is not None:
            for task, answer in exploration:
                lines.append(f"| {task} | {answer if name.startswith('VSF') else 'Cannot be answered from this system'} |")
        lines.append("")
        lines.append(f"Conditions spent: {sum(len(g.conditions) for g in groups)}; rows in highlighted groups: {covered}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--d", type=int, default=3, help="dimensionality of the VSF branch shown (the 3D cube by default)")
    ap.add_argument("--tau2", type=float, default=None, help="boundary of task 12 (default: tau + 0.1)")
    ap.add_argument("--title", default=None, help="heading of the two competitor screens")
    args = ap.parse_args()

    frame, entry = load_dataset(args.dataset)
    feats = [c for c in frame.columns if c != entry.target]
    X = frame[feats].values
    z = (frame[entry.target].values == args.positive).astype(np.int64)
    if not z.any():
        raise SystemExit(f"{args.positive!r} does not occur in column {entry.target!r}")
    schema, vsf = vsf_groups(X, z, feats, args.tau, args.m, args.d)
    if not vsf:
        raise SystemExit("the VSF branch has no centres at these settings")
    budget = sum(len(g.conditions) for g in vsf)
    rules = rule_groups(X, z, feats, args.tau, args.m, budget)
    nodes, tree, leaves = tree_groups(X, z, feats, args.tau, args.m, budget)
    cells = grid_cells(X, z, feats, schema)
    if len(cells) > len(z) // 10:
        raise SystemExit("the schema's grid exceeds the capacity rule (N/10 cells): the app merges levels and these cells would not match it")
    systems = {"VSF grid (partial centres)": vsf, "Heat map (same cells and groups as the grid)": vsf, "Decision tree": tree, "Rule list": rules}
    partitions = {"VSF grid (partial centres)": cells, "Heat map (same cells and groups as the grid)": cells, "Decision tree": leaves}
    # the cases are chosen from the systems that show every row (the rule list is an extra screen)
    cases = pick_cases({"grid": vsf, "tree": tree}, len(z))

    out = ROOT / "experiments" / "user_study" / "out" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or f"Groups in the {args.dataset} data"
    header = _subtitle(title, entry.target, args.positive, len(z), int(z.sum()), args.tau, args.m)
    (out / "rule_list.html").write_text(rule_list_html(rules, header, entry.target, args.positive), encoding="utf-8")
    # the cleanest display comparison: the SAME groups as the grid, written as a list
    (out / "rule_list_same_groups_as_grid.html").write_text(rule_list_html(vsf, header, entry.target, args.positive), encoding="utf-8")
    (out / "tree.html").write_text(tree_html(nodes, args.tau, args.m, header, entry.target, args.positive), encoding="utf-8")
    settings = (f"Dataset `{args.dataset}` ({len(z)} rows), target `{entry.target} = {args.positive}` ({int(z.sum())} rows), "
                f"purity floor {args.tau}, minimum group size {args.m}, observed-purity rule, all rows. "
                f"VSF: partial-axis centres, {args.d}D branch. Rule list and tree: condition budget {budget} (what the VSF branch spends).")
    (out / "heatmap.html").write_text(heatmap_html(cells, vsf, schema, header, entry.target, args.positive), encoding="utf-8")
    tau2 = args.tau2 if args.tau2 is not None else round(args.tau + 0.1, 2)
    exploration = exploration_answers(X, z, feats, args.tau, args.m, args.d, tau2)
    key = answer_key(systems, X, feats, cases, title, settings, schema, partitions, args.m, exploration)
    (out / "answer_key.md").write_text(key, encoding="utf-8")
    print(key)
    print(f"wrote {out}/heatmap.html, tree.html, rule_list.html, rule_list_same_groups_as_grid.html, answer_key.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
