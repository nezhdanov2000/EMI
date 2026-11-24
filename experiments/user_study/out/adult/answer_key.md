# Answer key: People who earn more than 50K a year

Dataset `adult` (48842 rows), target `income = >50K` (11687 rows), purity floor 0.8, minimum group size 100, observed-purity rule, all rows. VSF: partial-axis centres, 3D branch. Rule list and tree: condition budget 9 (what the VSF branch spends).

## Case cards (print on the system sheet)

- **Case 1** (row 10 of the file): education = Bachelors; marital = Married-civ-spouse; occupation = Exec-managerial; relationship = Husband; sex = Male; capital_gain = yes; capital_loss = no; hours = hours_q0
- **Case 2** (row 1 of the file): education = Bachelors; marital = Never-married; occupation = Adm-clerical; relationship = Not-in-family; sex = Male; capital_gain = yes; capital_loss = no; hours = hours_q0
- **Case 3** (row 9 of the file): education = Masters; marital = Never-married; occupation = Prof-specialty; relationship = Not-in-family; sex = Female; capital_gain = yes; capital_loss = no; hours = hours_q2

## VSF grid (partial centres)

Schema on screen: education x marital x capital_gain

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | education = Bachelors AND marital = Married-civ-spouse AND capital_gain = yes | 650 | 580 | 89% |
| 2 | education = Prof-school AND marital = Married-civ-spouse | 596 | 504 | 85% |
| 3 | education = Masters AND capital_gain = yes | 407 | 342 | 84% |
| 4 | education = Doctorate AND marital = Married-civ-spouse | 403 | 332 | 82% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 4 |
| 2. Columns used | education, marital, capital_gain |
| 3. Largest group | education = Bachelors AND marital = Married-civ-spouse AND capital_gain = yes (650 rows) |
| 4. Highest target share | education = Bachelors AND marital = Married-civ-spouse AND capital_gain = yes (89%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 46% (74 of 161; education = Bachelors AND marital = Never-married AND capital_gain = yes) |
| 10. Not highlighted, at least 100 rows, highest share | education = Some-college AND marital = Married-civ-spouse AND capital_gain = yes: 73% (327 of 445) |
| 11. Largest group shown, highlighted or not | education = HS-grad AND marital = Married-civ-spouse AND capital_gain = no: 6575 rows |
| 12. Coverage on the 3D card after the boundary is moved to 90% | 7.8% (2 centres; schema occupation + relationship + capital_gain) |
| 13. Card with the highest coverage at the original boundary | 4D: 18.9%, 6 centres, 15 conditions |
| 14. Interval of centre #1 on the 3D card | 89.2% [84.4, 93.0]; lower bound at or above the boundary: Yes |

Conditions spent: 9; rows in highlighted groups: 2056.

## Heat map (same cells and groups as the grid)

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | education = Bachelors AND marital = Married-civ-spouse AND capital_gain = yes | 650 | 580 | 89% |
| 2 | education = Prof-school AND marital = Married-civ-spouse | 596 | 504 | 85% |
| 3 | education = Masters AND capital_gain = yes | 407 | 342 | 84% |
| 4 | education = Doctorate AND marital = Married-civ-spouse | 403 | 332 | 82% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 4 |
| 2. Columns used | education, marital, capital_gain |
| 3. Largest group | education = Bachelors AND marital = Married-civ-spouse AND capital_gain = yes (650 rows) |
| 4. Highest target share | education = Bachelors AND marital = Married-civ-spouse AND capital_gain = yes (89%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 46% (74 of 161; education = Bachelors AND marital = Never-married AND capital_gain = yes) |
| 10. Not highlighted, at least 100 rows, highest share | education = Some-college AND marital = Married-civ-spouse AND capital_gain = yes: 73% (327 of 445) |
| 11. Largest group shown, highlighted or not | education = HS-grad AND marital = Married-civ-spouse AND capital_gain = no: 6575 rows |
| 12. Coverage on the 3D card after the boundary is moved to 90% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 9; rows in highlighted groups: 2056.

## Decision tree

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | marital = Married-civ-spouse AND capital_gain ≠ no AND education ≠ HS-grad | 2015 | 1619 | 80% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 1 |
| 2. Columns used | marital, capital_gain, education |
| 3. Largest group | marital = Married-civ-spouse AND capital_gain ≠ no AND education ≠ HS-grad (2015 rows) |
| 4. Highest target share | marital = Married-civ-spouse AND capital_gain ≠ no AND education ≠ HS-grad (80%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | No |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 26% (225 of 854; marital ≠ Married-civ-spouse AND capital_gain = yes AND hours = hours_q0) |
| 10. Not highlighted, at least 100 rows, highest share | marital = Married-civ-spouse AND capital_gain = no AND education = Bachelors: 63% (2200 of 3486) |
| 11. Largest group shown, highlighted or not | marital ≠ Married-civ-spouse AND capital_gain ≠ yes AND hours = hours_q0: 19943 rows |
| 12. Coverage on the 3D card after the boundary is moved to 90% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 3; rows in highlighted groups: 2015.

## Rule list

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | occupation = Exec-managerial AND capital_gain = yes | 810 | 660 | 81% |
| 2 | occupation = Prof-specialty AND sex = Male AND capital_gain = yes | 592 | 513 | 87% |
| 3 | relationship = Husband AND capital_loss = yes AND hours = hours_q2 | 500 | 402 | 80% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 3 |
| 2. Columns used | occupation, capital_gain, sex, relationship, capital_loss, hours |
| 3. Largest group | occupation = Exec-managerial AND capital_gain = yes (810 rows) |
| 4. Highest target share | occupation = Prof-specialty AND sex = Male AND capital_gain = yes (87%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | No |
| 8. Can one row be in two groups? | No |
| 12. Coverage on the 3D card after the boundary is moved to 90% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 8; rows in highlighted groups: 1902.
