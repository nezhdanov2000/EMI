# Answer key: Employees who left the company

Dataset `attrition` (1470 rows), target `attrition = Yes` (237 rows), purity floor 0.5, minimum group size 20, observed-purity rule, all rows. VSF: partial-axis centres, 3D branch. Rule list and tree: condition budget 10 (what the VSF branch spends).

## Case cards (print on the system sheet)

- **Case 1** (row 3 of the file): businesstravel = Travel_Rarely; joblevel = 1; jobrole = Laboratory_Technician; maritalstatus = Single; overtime = Yes; stockoptionlevel = 0; monthlyincome = MonthlyIncome_q0; yearsatcompany = YearsAtCompany_q0
- **Case 2** (row 2 of the file): businesstravel = Travel_Frequently; joblevel = 2; jobrole = Research_Scientist; maritalstatus = Married; overtime = No; stockoptionlevel = 1; monthlyincome = MonthlyIncome_q2; yearsatcompany = YearsAtCompany_q3
- **Case 3** (row 1 of the file): businesstravel = Travel_Rarely; joblevel = 2; jobrole = Sales_Executive; maritalstatus = Single; overtime = Yes; stockoptionlevel = 0; monthlyincome = MonthlyIncome_q2; yearsatcompany = YearsAtCompany_q2

## VSF grid (partial centres)

Schema on screen: jobrole x maritalstatus x overtime

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | jobrole = Laboratory_Technician AND overtime = Yes | 62 | 31 | 50% |
| 2 | jobrole = Sales_Representative AND maritalstatus = Single | 38 | 21 | 55% |
| 3 | jobrole = Sales_Executive AND maritalstatus = Single AND overtime = Yes | 32 | 20 | 62% |
| 4 | jobrole = Research_Scientist AND maritalstatus = Single AND overtime = Yes | 35 | 18 | 51% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 4 |
| 2. Columns used | jobrole, overtime, maritalstatus |
| 3. Largest group | jobrole = Laboratory_Technician AND overtime = Yes (62 rows) |
| 4. Highest target share | jobrole = Sales_Executive AND maritalstatus = Single AND overtime = Yes (62%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 2% (2 of 80; jobrole = Research_Scientist AND maritalstatus = Married AND overtime = No) |
| 10. Not highlighted, at least 20 rows, highest share | jobrole = Research_Scientist AND maritalstatus = Married AND overtime = Yes: 29% (12 of 42) |
| 11. Largest group shown, highlighted or not | jobrole = Sales_Executive AND maritalstatus = Married AND overtime = No: 115 rows |
| 12. Coverage on the 3D card after the boundary is moved to 60% | 26.2% (1 centres; schema joblevel + overtime + monthlyincome) |
| 13. Card with the highest coverage at the original boundary | 4D: 44.3%, 2 centres, 5 conditions |
| 14. Interval of centre #1 on the 3D card | 50.0% [32.3, 67.7]; lower bound below the boundary: No |

Conditions spent: 10; rows in highlighted groups: 167.

## Heat map (same cells and groups as the grid)

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | jobrole = Laboratory_Technician AND overtime = Yes | 62 | 31 | 50% |
| 2 | jobrole = Sales_Representative AND maritalstatus = Single | 38 | 21 | 55% |
| 3 | jobrole = Sales_Executive AND maritalstatus = Single AND overtime = Yes | 32 | 20 | 62% |
| 4 | jobrole = Research_Scientist AND maritalstatus = Single AND overtime = Yes | 35 | 18 | 51% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 4 |
| 2. Columns used | jobrole, overtime, maritalstatus |
| 3. Largest group | jobrole = Laboratory_Technician AND overtime = Yes (62 rows) |
| 4. Highest target share | jobrole = Sales_Executive AND maritalstatus = Single AND overtime = Yes (62%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 2% (2 of 80; jobrole = Research_Scientist AND maritalstatus = Married AND overtime = No) |
| 10. Not highlighted, at least 20 rows, highest share | jobrole = Research_Scientist AND maritalstatus = Married AND overtime = Yes: 29% (12 of 42) |
| 11. Largest group shown, highlighted or not | jobrole = Sales_Executive AND maritalstatus = Married AND overtime = No: 115 rows |
| 12. Coverage on the 3D card after the boundary is moved to 60% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 10; rows in highlighted groups: 167.

## Decision tree

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | overtime = Yes AND joblevel = 1 | 156 | 82 | 53% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 1 |
| 2. Columns used | overtime, joblevel |
| 3. Largest group | overtime = Yes AND joblevel = 1 (156 rows) |
| 4. Highest target share | overtime = Yes AND joblevel = 1 (53%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | No |
| 8. Can one row be in two groups? | No |
| 9. Share in the group of Case 2 | 6% (38 of 605; overtime ≠ Yes AND stockoptionlevel ≠ 0) |
| 10. Not highlighted, at least 20 rows, highest share | overtime = Yes AND joblevel ≠ 1: 17% (45 of 260) |
| 11. Largest group shown, highlighted or not | overtime ≠ Yes AND stockoptionlevel ≠ 0: 605 rows |
| 12. Coverage on the 3D card after the boundary is moved to 60% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 2; rows in highlighted groups: 156.

## Rule list

| # | Group | Rows | With target | Share |
| --- | --- | --- | --- | --- |
| 1 | joblevel = 1 AND overtime = Yes | 156 | 82 | 53% |
| 2 | jobrole = Sales_Executive AND overtime = Yes AND stockoptionlevel = 0 | 42 | 23 | 55% |
| 3 | businesstravel = Travel_Frequently AND overtime = No AND monthlyincome = MonthlyIncome_q0 AND yearsatcompany = YearsAtCompany_q0 | 27 | 14 | 52% |

| Task | Correct answer |
| --- | --- |
| 1. Number of highlighted groups | 3 |
| 2. Columns used | joblevel, overtime, jobrole, stockoptionlevel, businesstravel, monthlyincome, yearsatcompany |
| 3. Largest group | joblevel = 1 AND overtime = Yes (156 rows) |
| 4. Highest target share | jobrole = Sales_Executive AND overtime = Yes AND stockoptionlevel = 0 (55%) |
| 5. Case 1 | Yes |
| 6. Case 2 | No |
| 7. Case 3 | Yes |
| 8. Can one row be in two groups? | No |
| 12. Coverage on the 3D card after the boundary is moved to 60% | Cannot be answered from this system |
| 13. Card with the highest coverage at the original boundary | Cannot be answered from this system |
| 14. Interval of centre #1 on the 3D card | Cannot be answered from this system |

Conditions spent: 9; rows in highlighted groups: 225.
