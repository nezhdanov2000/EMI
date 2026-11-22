# Experiment Execution and Paper Revision Report (2026-10-19 Review)

This report details the execution and results of the remaining experiments for the paper revision, including per-job computation times, splits computed, DL8.5 timeout profiles, every changed sentence with recomputed numbers, and qualitative statement updates.

---

## 1. Per-Job Wall-Clock Time and Splits Computed

All experiments were executed with up to 12 concurrent workers on a 16-core machine, resuming from and updating the per-split cache.

### Job A: Optimal Tree on the Two Wide Tables (`adult` and `connect_4`)
- **`adult` (`tree_opt`)**:
  - **Splits computed**: 30 splits total
    - `purity` ($\tau=0.7, m=20$): 10 splits ($2 \text{ repeats} \times 5 \text{ folds}$)
    - `certified` ($\tau=0.7, m=20$): 10 splits ($2 \text{ repeats} \times 5 \text{ folds}$)
    - `split` ($\tau=0.7, m=20$): 10 splits ($2 \text{ repeats} \times 5 \text{ folds}$)
  - **Computation time**:
    - `purity`: $3,705.3\text{ s}$
    - `certified`: $66,634.5\text{ s}$
    - `split`: $2,769.0\text{ s}$
    - **Total cumulative compute**: $73,108.8\text{ s}$ (~$20.3\text{ hours}$)
    - **Wall-clock time**: ~1.9 hours (parallelized across splits)
- **`connect_4` (`tree_opt`)**:
  - **Splits computed**: 30 splits total
    - `loss purity` ($\tau=0.7, m=20$): 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$)
    - `loss certified` ($\tau=0.7, m=20$): 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$)
    - `loss split` ($\tau=0.7, m=20$): 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$)
    - `win purity` ($\tau=0.9, m=20$): 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$)
    - `win certified` ($\tau=0.9, m=20$): 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$)
    - `win split` ($\tau=0.9, m=20$): 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$)
  - **Computation time**:
    - `loss purity`: $42,500.7\text{ s}$
    - `loss certified`: $55,837.1\text{ s}$
    - `loss split`: $17,570.9\text{ s}$
    - `win purity`: $27,176.2\text{ s}$
    - `win certified`: $39,683.8\text{ s}$
    - `win split`: $21,101.9\text{ s}$
    - **Total cumulative compute**: $203,870.6\text{ s}$ (~$56.6\text{ hours}$)
    - **Wall-clock time**: ~5.1 hours (parallelized across splits and configurations)

---

### Job B: WRAcc Lists on `connect_4` (`wracc`, `wracc_raw`)
- **Splits computed**: 30 splits total across 6 configurations ($1 \text{ repeat} \times 5 \text{ folds}$ each):
  - `loss purity`, `loss certified`, `loss split`
  - `win purity`, `win certified`, `win split`
- **Execution details**: Memory-managed execution to prevent peak memory starvation from 42 columns $\times$ 54,000 rows during the in-house rule enumeration (`baselines._rule_candidates`).
- **Computation time**: ~59,187 s cumulative across configurations.
- **Wall-clock time**: ~1.6 hours.

---

### Job C: Holdout Split on the Four Certified Configurations Lacking It
Run with `--selection split`:
1. **`car_evaluation`**: 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$); compute time: $97.7\text{ s}$; wall-clock: ~25 s.
2. **`resume`**: 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$); compute time: $3,025.8\text{ s}$; wall-clock: ~7 min.
3. **`breast_cancer`**: 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$); compute time: $1,126.6\text{ s}$; wall-clock: ~3 min.
4. **`splice`**: 5 splits ($1 \text{ repeat} \times 5 \text{ folds}$); subsumed into the 15-split rerun in Job D.

---

### Job D: More Repeats on `chess_krkp` and `splice` (`n_repeats=3`)
Both tables recomputed with $n\_repeats=3$ ($3 \text{ repeats} \times 5 \text{ folds} = 15\text{ splits}$) across all 3 selections (`purity`, `certified`, `split`) for all comparison methods (`vsf`, `vsf_partial`, `vsf_greedy`, `rules`, `rules_pure`, `rules_disjoint`, `tree`, `ssdpp`, `wracc`, `wracc_raw`, `tree_opt`):
- **`chess_krkp`**:
  - **Splits computed**: 45 splits total (15 purity, 15 certified, 15 split)
  - **Computation time**:
    - `purity`: $48,619.5\text{ s}$
    - `certified`: $14,257.4\text{ s}$
    - `split`: $5,050.2\text{ s}$
    - **Total cumulative compute**: $67,927.1\text{ s}$ (~$18.9\text{ hours}$)
  - **Wall-clock time**: ~1.8 hours.
- **`splice`**:
  - **Splits computed**: 45 splits total (15 purity, 15 certified, 15 split)
  - **Computation time**:
    - `purity`: $2,054.8\text{ s}$
    - `certified`: $3,563.3\text{ s}$
    - `split`: $2,642.6\text{ s}$
    - **Total cumulative compute**: $8,260.7\text{ s}$ (~$2.3\text{ hours}$)
  - **Wall-clock time**: ~22 min.

---

## 2. DL8.5 Fits that Hit the Time Limit

DL8.5 has a $600\text{ s}$ per-depth limit (`baselines.TREE_OPT_TIME_LIMIT = 600`), returning the best tree found if the time limit is reached.

As reported in regenerated Table 10 (`paper/tables/tree_opt_B32.tex`, `tab:treeopt`):
- **`adult`**:
  - `purity` ($\tau=0.7$): **0% timeouts** (all fits completed strictly within $600\text{ s}$).
  - `certified` ($\tau=0.7$): **0% timeouts** (all fits completed strictly within $600\text{ s}$).
- **`connect_4`**:
  - `purity`, `loss` ($\tau=0.7$): **100% timeouts** (5 of 5 splits hit the $600\text{ s}$ limit at depth 4).
  - `purity`, `win` ($\tau=0.9$): **40% timeouts** (2 of 5 splits hit the $600\text{ s}$ limit at depth 4).
  - `certified`, `loss` ($\tau=0.7$): **80% timeouts** (4 of 5 splits hit the $600\text{ s}$ limit at depth 4).
  - `certified`, `win` ($\tau=0.9$): **100% timeouts** (5 of 5 splits hit the $600\text{ s}$ limit at depth 4).
- **All other 15 benchmark tables**: **0% timeouts** (100% optimal depth-4 trees proven within time limit).

---

## 3. Every Changed Sentence (Old -> New Numbers)

Each number was updated strictly from the regenerated CSV files (`pairwise_partial.csv`, `comparison_summary.csv`, and LaTeX tables) and verified with `experiments/check_numbers.py` (53 statements found, 0 missing).

1. **Holm surviving wins against full grid under certificate** (Section 7.1, Line 212):
   - *Old*: `(39 and 12 wins survive)`
   - *New*: `(39 and 15 wins survive)`

2. **Net coverage mean for tree under certificate** (Section 7.4, Line 229):
   - *Old*: `and $0.19$ for the tree`
   - *New*: `and $0.18$ for the tree`

3. **Greedy chain mean gap at $B=32$** (Section 7.4, Line 236):
   - *Old*: `(mean gap at $B=32$: 6.2 and 3.4 points)`
   - *New*: `(mean gap at $B=32$: 6.3 and 3.4 points)`

4. **SSD++ losses in observed purity** (Section 7.6, Line 246):
   - *Old*: `beat it in 57 of 109 observed-purity cells and lose 15`
   - *New*: `beat it in 57 of 109 observed-purity cells and lose 16`

5. **Compactness of overlapping rules** (Section 7.2, Line 222):
   - *Old*: `overlapping rules reach $0.58$ by naming ten columns and spending $26$ conditions`
   - *New*: `overlapping rules reach $0.59$ by naming ten columns and spending $27$ conditions`

6. **Optimal tree counts and mean coverage** (Section 7.5, Line 240):
   - *Old*: `partial centres beat it in 27 of 93 cells and lose 12 (at $B=32$: $0.49$ against $0.47$ mean coverage); under the certificate they win 8 and lose 10 ($0.50$ against $0.51$)`
   - *New*: `partial centres beat it in 33 of 109 cells and lose 13 (at $B=32$: $0.46$ against $0.43$ mean coverage); under the certificate they win 14 and lose 15 ($0.40$ against $0.40$)`

7. **Optimal tree net coverage** (Section 7.5, Line 240):
   - *Old*: `($0.22$ against $0.15$ and $0.43$ against $0.36$ at $B=32$)`
   - *New*: `($0.20$ against $0.14$ and $0.33$ against $0.28$ at $B=32$)`

8. **Optimal tree vs CART table counts** (Section 7.5, Line 240):
   - *Old*: `covers more on six tables at $B=32$ and less on six`
   - *New*: `covers more on nine tables at $B=32$ and less on six`

9. **WRAcc vs max coverage rules at $B=32$** (Section 7.7, Line 243):
   - *Old*: `($0.49$ against $0.60$ at $B=32$ under the observed rule)`
   - *New*: `($0.46$ against $0.59$ at $B=32$ under the observed rule)`

10. **WRAcc held-out purity share** (Section 7.7, Line 243):
    - *Old*: `reaches $\tau$ in $68\,\%$ of the splits against $58\,\%$`
    - *New*: `reaches $\tau$ in $71\,\%$ of the splits against $60\,\%$`

11. **Partial centres vs WRAcc counts and purity** (Section 7.7, Line 243):
    - *Old*: `($0.49$ against $0.49$; partial centres win 22 of 99 cells and lose 7, under the certificate 10 and 1) with lower held-out purity ($68$ against $77\,\%$ of splits)`
    - *New*: `($0.46$ against $0.46$; partial centres win 27 of 109 cells and lose 7, under the certificate 10 and 1) with lower held-out purity ($71$ against $79\,\%$ of splits)`

12. **WRAcc_raw coverage and purity share** (Section 7.7, Line 243):
    - *Old*: `($0.87$ at $B=32$, beating partial centres in 75 of 99 cells) with a union that reaches $\tau$ in $18\,\%$ of the splits`
    - *New*: `($0.83$ at $B=32$, beating partial centres in 85 of 109 cells) with a union that reaches $\tau$ in $16\,\%$ of the splits`

13. **WRAcc_raw zero-purity share configurations** (Section 7.7, Line 243):
    - *Old*: `in none of them on 16 of the 19 configurations at $B=32$`
    - *New*: `in none of them on 18 of the 21 configurations at $B=32$`

14. **Holdout split comparison on common certified configurations** (Section 4, Line 139):
    - *Old*: `mean $30$ against $40$ points`
    - *New*: `mean $31$ against $40$ points`

---

## 4. Textual and Structural Edits Made

1. **Section 5 (tree_opt description)**:
   - Replaced caveat sentence with:
     > `Run on all 17 tables; on \texttt{adult} no fit hit the ten-minute limit (0\,\% timeouts in Table~\ref{tab:treeopt}), while on \texttt{connect\_4} fits hit the limit on 40--100\,\% of splits (Table~\ref{tab:treeopt}). No fit on the other tables hit the ten-minute limit.`
2. **Caveat Removals across Sections**:
   - Removed "on the 14 tables where the tree was run" / "14 of 17 tables" caveats from Introduction (Contribution 2), Section 7.5, Section 7.6, and Conclusion.
   - Removed the caveat about excluding the two widest tables.
3. **Section 7.7**:
   - Deleted the concluding caveat: `\texttt{connect\_4} is not in these counts (the rule enumeration on $54\,000$ rows and 42 columns was not repeated for the two lists).`
4. **Folds and Degrees of Freedom (Section 5 & Limitations)**:
   - Updated fold definitions: `chess` and `splice` have $5 \times 3 = 15$ folds; `connect_4` keeps $5 \times 1 = 5$ folds.
   - Updated marks degrees of freedom: 4 on `connect_4`, 14 on `chess` and `splice`.
5. **Limitations**:
   - Deleted clauses stating that optimal trees were not run on `adult` and `connect_4`, that WRAcc lists were not run on `connect_4`, and that holdout splits were run on only 15 of 19 configurations.

---

## 5. Qualitative Statements Contradicted / Affected by New Numbers

1. **Optimal Tree vs CART at Budget $B=32$**:
   - *Previous claim*: Optimal tree and CART tied across benchmark datasets ("covers more on six tables at $B=32$ and less on six").
   - *Recomputed finding*: Optimal tree now holds an advantage, covering more than CART on **nine** tables and less on **six** (a +3 net dataset advantage).
2. **Partial Centres vs Optimal Tree under Certificate at $B=32$**:
   - *Previous claim*: Partial centres lost to optimal tree on mean certified coverage ($0.50$ against $0.51$).
   - *Recomputed finding*: Partial centres now **tie** optimal tree on mean certified coverage ($0.40$ against $0.40$).
3. **Certified Significance vs Full Grid (Holm Adjustment)**:
   - Significant wins surviving Holm correction increased from **12** to **15** wins out of 53 certified cells.
4. **SSD++ Losses under Observed Purity**:
   - Increased slightly from **15** to **16** cell losses (out of 109 cells).

---

## 6. Verification and Build Status

- `experiments/check_numbers.py`: **53 statements found, 0 missing** (passed with code 0).
- LaTeX build (`pdflatex` $\times 3$ + `bibtex`): **Built cleanly** to `paper/main.pdf` (24 pages, code 0).
- Test suite (`python -m pytest -q`): **483 passed, 1 skipped in 222.76s** (passed with code 0).
- Git repository: Clean uncommitted state preserved; **no commits made**.


---

## Corrections after review (Claude, 2026-10-20)

* `compare_adult.csv` and `compare_connect_4.csv` had been written by `--methods tree_opt` / `--methods wracc` runs
  and lacked the other extra methods; both were regenerated from the complete caches, then
  `summarize_comparison.py`, `paper_tables.py`, `compactness.py` and `check_numbers.py` (53/53) were rerun.
* `compare_baselines._save` now raises after ten failed renames instead of dropping the result silently;
  `baselines.summarize_splits` prints the methods it leaves out when they are missing on some splits.
* Text: splice paragraph in Section 7.8 and Limitations rewritten for the 15-split numbers (family empty on
  9 of 15 folds, grid/rules 4-5 %, SSD++ 19 % at 0.83); Section 7.6 states that on connect_4 the DL8.5 tree is
  the anytime result at the 600 s limit; Section 7.10 (Cost) says the chess timing is from the single-repeat run.
* Method names in Job D above were wrong in the first version of this report (there is no cart/cn2/ripper
  baseline); corrected.
