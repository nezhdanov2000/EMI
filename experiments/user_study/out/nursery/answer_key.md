# Answer key: Nursery applications ranked as priority

Dataset `nursery` (12960 rows), target `class = priority` (4266 rows), purity floor 0.7, minimum group size 20, observed-purity rule, all rows. VSF: partial-axis centres, 3D branch. Rule list and tree: condition budget 20 (what the VSF branch spends).

## Case cards (print on the system sheet)

- **Case 1** (row 1 of the file): parents = usual; has_nurs = proper; health = recommended
- **Case 2** (row 3 of the file): parents = usual; has_nurs = proper; health = not_recom
- **Case 3** (row 2593 of the file): parents = usual; has_nurs = critical; health = recommended

## VSF grid (partial centres)

Schema on screen: parents x has_nurs x health

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | has_nurs = less_proper AND health = priority | 864 | 675 | 78% |
| 2 | has_nurs = proper AND health = priority | 864 | 675 | 78% |
| 3 | has_nurs = less_proper AND health = recommended | 864 | 669 | 77% |
| 4 | has_nurs = proper AND health = recommended | 864 | 669 | 77% |
| 5 | parents = usual AND has_nurs = improper AND health = priority | 288 | 288 | 100% |
| 6 | parents = pretentious AND has_nurs = improper AND health = recommended | 288 | 225 | 78% |
| 7 | parents = usual AND has_nurs = critical AND health = recommended | 288 | 225 | 78% |
| 8 | parents = usual AND has_nurs = improper AND health = recommended | 288 | 222 | 77% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 8 |
| 2. Columns used | has_nurs, health, parents |
| 3. Largest group | has_nurs = less_proper AND health = priority (864 rows) |
| 4. Highest target share | parents = usual AND has_nurs = improper AND health = priority (100%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 0% (0 of 288; parents = usual AND has_nurs = proper AND health = not_recom) |
| 10. Not highlighted, at least 20 rows, highest share | parents = pretentious AND has_nurs = improper AND health = priority: 34% (99 of 288) [2 tied] |
| 11. Largest group shown, highlighted or not | parents = great_pret AND has_nurs = critical AND health = not_recom: 288 rows |
| 12. Coverage on the 3D card after the boundary is moved to 80% | 33.9% (8 centres; schema has_nurs + children + health) |
| 13. Card with the highest coverage at the original boundary | 3D: 85.5%, 8 centres, 20 conditions |
| 14. Interval of centre #1 on the 3D card | 78.1% [74.1, 81.8]; lower bound at or above the boundary: Yes |

Conditions spent: 20; rows in highlighted groups: 4608.

## Heat map (same cells and groups as the grid)

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | has_nurs = less_proper AND health = priority | 864 | 675 | 78% |
| 2 | has_nurs = proper AND health = priority | 864 | 675 | 78% |
| 3 | has_nurs = less_proper AND health = recommended | 864 | 669 | 77% |
| 4 | has_nurs = proper AND health = recommended | 864 | 669 | 77% |
| 5 | parents = usual AND has_nurs = improper AND health = priority | 288 | 288 | 100% |
| 6 | parents = pretentious AND has_nurs = improper AND health = recommended | 288 | 225 | 78% |
| 7 | parents = usual AND has_nurs = critical AND health = recommended | 288 | 225 | 78% |
| 8 | parents = usual AND has_nurs = improper AND health = recommended | 288 | 222 | 77% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 8 |
| 2. Columns used | has_nurs, health, parents |
| 3. Largest group | has_nurs = less_proper AND health = priority (864 rows) |
| 4. Highest target share | parents = usual AND has_nurs = improper AND health = priority (100%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 0% (0 of 288; parents = usual AND has_nurs = proper AND health = not_recom) |
| 10. Not highlighted, at least 20 rows, highest share | parents = pretentious AND has_nurs = improper AND health = priority: 34% (99 of 288) [2 tied] |
| 11. Largest group shown, highlighted or not | parents = great_pret AND has_nurs = critical AND health = not_recom: 288 rows |
| 12. Coverage on the 3D card after the boundary is moved to 80% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 20; rows in highlighted groups: 4608.

## Decision tree

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | health ≠ not_recom AND has_nurs ≠ very_crit AND has_nurs ≠ critical AND parents ≠ great_pret | 3456 | 2874 | 83% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 1 |
| 2. Columns used | health, has_nurs, parents |
| 3. Largest group | health ≠ not_recom AND has_nurs ≠ very_crit AND has_nurs ≠ critical AND parents ≠ great_pret (3456 rows) |
| 4. Highest target share | health ≠ not_recom AND has_nurs ≠ very_crit AND has_nurs ≠ critical AND parents ≠ great_pret (83%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | No |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 0% (0 of 4320; health = not_recom) |
| 10. Not highlighted, at least 20 rows, highest share | health ≠ not_recom AND has_nurs ≠ very_crit AND has_nurs = critical AND parents = usual: 56% (324 of 576) |
| 11. Largest group shown, highlighted or not | health = not_recom: 4320 rows |
| 12. Coverage on the 3D card after the boundary is moved to 80% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 4; rows in highlighted groups: 3456.

## Rule list

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | has_nurs = less_proper AND health = priority | 864 | 675 | 78% |
| 2 | has_nurs = proper AND health = priority | 864 | 675 | 78% |
| 3 | has_nurs = less_proper AND health = recommended | 864 | 669 | 77% |
| 4 | has_nurs = proper AND health = recommended | 864 | 669 | 77% |
| 5 | parents = usual AND has_nurs = improper AND health = priority | 288 | 288 | 100% |
| 6 | parents = pretentious AND has_nurs = improper AND health = recommended | 288 | 225 | 78% |
| 7 | parents = usual AND has_nurs = critical AND health = recommended | 288 | 225 | 78% |
| 8 | parents = usual AND has_nurs = improper AND health = recommended | 288 | 222 | 77% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 8 |
| 2. Columns used | has_nurs, health, parents |
| 3. Largest group | has_nurs = less_proper AND health = priority (864 rows) |
| 4. Highest target share | parents = usual AND has_nurs = improper AND health = priority (100%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 12. Coverage on the 3D card after the boundary is moved to 80% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 20; rows in highlighted groups: 4608.
