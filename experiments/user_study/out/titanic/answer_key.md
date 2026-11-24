# Answer key: Practice: Titanic passengers who survived

Dataset `titanic` (891 rows), target `survived = survived` (342 rows), purity floor 0.7, minimum group size 5, observed-purity rule, all rows. VSF: partial-axis centres, 3D branch. Rule list and tree: condition budget 3 (what the VSF branch spends).

## Case cards (print on the system sheet)

- **Case 1** (row 2 of the file): passenger_class = 1st_Class; sex = female; title = Mrs; family_type = small_family
- **Case 2** (row 1 of the file): passenger_class = 3rd_Class; sex = male; title = Mr; family_type = small_family
- **Case 3** (row 3 of the file): passenger_class = 3rd_Class; sex = female; title = Miss; family_type = alone

## VSF grid (partial centres)

Schema on screen: sex x title x family_type

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | sex = female | 314 | 233 | 74% |
| 2 | title = Master AND family_type = small_family | 18 | 18 | 100% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 2 |
| 2. Columns used | sex, title, family_type |
| 3. Largest group | sex = female (314 rows) |
| 4. Highest target share | title = Master AND family_type = small_family (100%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 18% (19 of 103; sex = male AND title = Mr AND family_type = small_family) |
| 10. Not highlighted, at least 5 rows, highest share | sex = male AND title = Officer AND family_type = alone: 25% (3 of 12) |
| 11. Largest group shown, highlighted or not | sex = male AND title = Mr AND family_type = alone: 397 rows |
| 12. Coverage on the 3D card after the boundary is moved to 80% | 56.1% (6 centres; schema passenger_class + title + family_type) |
| 13. Card with the highest coverage at the original boundary | 4D: 74.9%, 3 centres, 8 conditions |
| 14. Interval of centre #1 on the 3D card | 74.2% [69.0, 79.0]; lower bound below the boundary: No |

Conditions spent: 3; rows in highlighted groups: 332.

## Heat map (same cells and groups as the grid)

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | sex = female | 314 | 233 | 74% |
| 2 | title = Master AND family_type = small_family | 18 | 18 | 100% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 2 |
| 2. Columns used | sex, title, family_type |
| 3. Largest group | sex = female (314 rows) |
| 4. Highest target share | title = Master AND family_type = small_family (100%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 18% (19 of 103; sex = male AND title = Mr AND family_type = small_family) |
| 10. Not highlighted, at least 5 rows, highest share | sex = male AND title = Officer AND family_type = alone: 25% (3 of 12) |
| 11. Largest group shown, highlighted or not | sex = male AND title = Mr AND family_type = alone: 397 rows |
| 12. Coverage on the 3D card after the boundary is moved to 80% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 3; rows in highlighted groups: 332.

## Decision tree

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | title ≠ Mr AND passenger_class ≠ 3rd_Class | 202 | 178 | 88% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 1 |
| 2. Columns used | title, passenger_class |
| 3. Largest group | title ≠ Mr AND passenger_class ≠ 3rd_Class (202 rows) |
| 4. Highest target share | title ≠ Mr AND passenger_class ≠ 3rd_Class (88%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | No |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 11% (47 of 424; title = Mr AND deck = unknown_deck) |
| 10. Not highlighted, at least 5 rows, highest share | title ≠ Mr AND passenger_class = 3rd_Class: 48% (83 of 172) |
| 11. Largest group shown, highlighted or not | title = Mr AND deck = unknown_deck: 424 rows |
| 12. Coverage on the 3D card after the boundary is moved to 80% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 2; rows in highlighted groups: 202.

## Rule list

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | sex = female | 314 | 233 | 74% |
| 2 | title = Master AND family_type = small_family | 18 | 18 | 100% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 2 |
| 2. Columns used | sex, title, family_type |
| 3. Largest group | sex = female (314 rows) |
| 4. Highest target share | title = Master AND family_type = small_family (100%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 12. Coverage on the 3D card after the boundary is moved to 80% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 3; rows in highlighted groups: 332.
