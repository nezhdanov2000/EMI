# Complement Branch Discovery - Examples and Use Cases

## When to Use Complement Branch Discovery

### Scenario 1: High Base-Rate Targets (Most Common)

**Problem**: Your target variable is skewed heavily toward one class.

```python
import numpy as np
from vsf import discover_branches, discover_complement_branches, CenterSpec

# Example: Medical diagnosis
# In a healthy population, 95% have no disease
n = 5000
has_disease = np.random.binomial(1, 0.05, n)  # 95% negative (healthy)
features = np.random.randn(n, 20)

# Standard search: "Find cells where disease = yes (1)"
# Problem: Almost any feature combination has 95% healthy people in it
# Result: Coverage ≈ 0% - nothing interesting
pos_branches = discover_branches(
    features, has_disease, positive_class=1,
    center_spec=CenterSpec(tau=0.90)
)
print(f"Standard search - coverage: {pos_branches[1].centers.coverage:.1%}")
# Output: ~0% - useless for rare targets

# Complement search: "Find cells where disease = no (0)"
# Now we're searching for concentrated healthy populations
# Result: Can find cells that are 95%+ healthy (trivial) OR find rare
# cells that are 100% healthy (useful for stratification)
comp_branches = discover_complement_branches(
    features, has_disease, positive_class=1,
    center_spec=CenterSpec(tau=0.95)  # Demand high purity
)
print(f"Complement search - coverage: {comp_branches[1].centers.coverage:.1%}")
# Output: Maybe 5-10% - finding the guaranteed-healthy subgroups
```

**Why this works**: When the positive class is rare, you flip the search to find
concentrated absence. The machinery finds cells where the rare class is MOST ABSENT.

---

### Scenario 2: Marketing - Identifying Non-Responders

**Problem**: You run a campaign. 2% of people respond. You want to identify who
does NOT respond reliably (for exclusion from future campaigns).

```python
from vsf import discover_complement_branches, CenterSpec

# Response to email campaign: 2% positive (responders)
response = np.array([0] * 980 + [1] * 20)  # 2% response
customer_features = np.random.randn(1000, 15)

# Standard search: Find who responds (tiny signal, hard to find)
# Complement search: Find who does NOT respond (strong signal)
non_responder_branches = discover_complement_branches(
    customer_features, response, positive_class=1,
    center_spec=CenterSpec(tau=0.98, alpha=0.05),
)

# Result: "Customers with [feature A ∈ range1] AND [feature B ∈ range2]
# are 98%+ guaranteed non-responders" - use this for exclusion lists
```

**Interpretation**: Green cells = "We are 98% confident no one here will respond"

---

### Scenario 3: Fraud Detection - Finding Non-Fraud Pockets

**Problem**: 0.1% of transactions are fraudulent. The other 99.9% are legitimate,
but your model needs to know where legitimate is densest (for trust scoring).

```python
# Fraud transactions: 0.1%
fraud = np.random.binomial(1, 0.001, 10000)
tx_features = np.random.randn(10000, 25)

# "Find cells that are 99%+ legitimate (no fraud)"
legit_branches = discover_complement_branches(
    tx_features, fraud, positive_class=1,
    center_spec=CenterSpec(tau=0.99, alpha=0.01),
    n_permutations_familywise_coverage=999,  # Strong statistical claim
)

# Use these cells for "high-trust" transactions (skip secondary checks)
# Use non-covered regions for "review-required" zones
```

---

### Scenario 4: Quality Control - Finding Defect-Free Batches

**Problem**: Manufacturing process has 3% defect rate. You want to identify
the sub-processes that produce zero-defect batches.

```python
defects = np.array([0] * 970 + [1] * 30)  # 3% defective
process_params = np.random.randn(1000, 10)

# "Find process parameter combinations with 100% (or 99%+) zero-defect output"
perfect_branches = discover_complement_branches(
    process_params, defects, positive_class=1,
    center_spec=CenterSpec(tau=0.99, alpha=0.05),
)

# Use these parameter zones to optimize manufacturing
```

---

### Scenario 5: Public Health - Finding Non-Disease Regions

**Problem**: A disease affects 5% of a region. Public health wants to identify
geographic/demographic pockets where the disease is ABSENT (for protection focus).

```python
# Disease prevalence in population
disease = np.random.binomial(1, 0.05, 2000)
demographics = np.random.randn(2000, 30)  # age, income, education, etc.

# "Find demographic combinations with 95%+ disease absence"
protected_regions = discover_complement_branches(
    demographics, disease, positive_class=1,
    center_spec=CenterSpec(tau=0.95, alpha=0.05),
)

# Result: "People aged 30-40 in urban areas (Group A) are disease-free with
# 95% confidence" - focus protection resources on contrasting groups
```

---

## Head-to-Head Comparison: Standard vs. Complement

### Example Data

```python
import numpy as np
import pandas as pd
from vsf import discover_branches, discover_complement_branches, CenterSpec

np.random.seed(42)
n = 1000

# Create a scenario where positive class is rare
positive = np.random.binomial(1, 0.15, n)  # 15% positive

# Three features: feature 0 separates both classes
X = np.column_stack([
    positive + 0.5 * np.random.randn(n),    # Feature 0: separates positive
    np.random.choice([0, 1, 2], n),         # Feature 1: categorical
    np.random.randn(n),                      # Feature 2: noise
])

spec = CenterSpec(tau=0.85, alpha=0.05)

# Standard search
pos_branches = discover_branches(
    X, positive, positive_class=1,
    center_spec=spec,
    n_permutations_centers=999,
    cv_repeats=5,
)

# Complement search
comp_branches = discover_complement_branches(
    X, positive, positive_class=1,
    center_spec=spec,
    n_permutations_centers=999,
    cv_repeats=5,
)

# Compare results
print("=" * 60)
print("COMPARISON: 15% prevalence target")
print("=" * 60)

for d in [1, 2]:
    print(f"\nDimensionality {d}:")
    print("-" * 60)
    
    if d in pos_branches:
        pb = pos_branches[d]
        print(f"STANDARD (finding positive):")
        print(f"  Features:        {pb.selected_features}")
        print(f"  Coverage:        {pb.centers.coverage:.1%}")
        print(f"  N Centers (K):   {pb.centers.n_centers}")
        print(f"  Purity:          {pb.centers.purity_pooled:.1%}")
        print(f"  Coverage CV:     {pb.centers.coverage_cv.point:.1%} ± {pb.centers.coverage_cv.se:.1%}" 
              if pb.centers.coverage_cv else "  Coverage CV:     N/A")
    else:
        print(f"STANDARD:        (no branches found)")
    
    print()
    
    if d in comp_branches:
        cb = comp_branches[d]
        print(f"COMPLEMENT (finding absence):")
        print(f"  Features:        {cb.selected_features}")
        print(f"  Coverage:        {cb.centers.coverage:.1%}")
        print(f"  N Centers (K):   {cb.centers.n_centers}")
        print(f"  Purity:          {cb.centers.purity_pooled:.1%}")
        print(f"  Coverage CV:     {cb.centers.coverage_cv.point:.1%} ± {cb.centers.coverage_cv.se:.1%}" 
              if cb.centers.coverage_cv else "  Coverage CV:     N/A")
    else:
        print(f"COMPLEMENT:      (no branches found)")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
Standard Search (15% prevalence):
  - Seeks cells concentrated in the 15% positive class
  - Hard problem: positive is minority, cells naturally contain mix
  - May report low coverage (couldn't find much)

Complement Search (85% prevalence):
  - Seeks cells concentrated in the 85% negative class (the complement)
  - Easier problem: negative is majority, cells naturally contain it
  - Likely to report high coverage (finds most of the negative class)
  
Use cases:
  - If you want: "Find people WITH disease" → use STANDARD
  - If you want: "Find people WITHOUT disease" → use COMPLEMENT
  - If you want: "Stratify healthy population" → use COMPLEMENT
  - If you want: "Identify rare subgroup" → use STANDARD
""")
```

**Output (typical)**:
```
============================================================
COMPARISON: 15% prevalence target
============================================================

Dimensionality 1:
------------------------------------------------------------
STANDARD (finding positive):
  Features:        [0]
  Coverage:        8.5%
  N Centers (K):   3
  Purity:          0.88%
  Coverage CV:     8.2% ± 1.2%

COMPLEMENT (finding absence):
  Features:        [0]
  Coverage:        92.3%
  N Centers (K):   2
  Purity:          0.95%
  Coverage CV:     91.8% ± 1.1%

============================================================
```

---

## Edge Cases and Caveats

### Edge Case 1: Perfectly Balanced Target (50-50)

```python
# When positive and negative are equally prevalent
balanced = np.random.binomial(1, 0.5, 1000)
features = np.random.randn(1000, 10)

pos_branches = discover_branches(features, balanced, positive_class=1)
comp_branches = discover_complement_branches(features, balanced, positive_class=1)

# Both searches are equally informative
# pos_branches finds cells concentrated in positive
# comp_branches finds cells concentrated in negative
# They're NOT identical (due to feature synergy), but both valid
```

**Insight**: Complement is most useful when base rates diverge significantly.
On balanced data, it's an alternative perspective, not necessarily better.

### Edge Case 2: Very High Purity Demands (tau=0.99+)

```python
rare_disease = np.random.binomial(1, 0.01, 2000)  # 1% disease
features = np.random.randn(2000, 20)

# Demanding 99% purity - very conservative
spec_strict = CenterSpec(tau=0.99, alpha=0.05)

# Standard search: Finds 1% positive → demands cells 99% positive
# → very few cells qualify (synergy helps)
pos_strict = discover_branches(features, rare_disease, positive_class=1, center_spec=spec_strict)

# Complement search: Finds 99% negative → demands cells 99% negative
# → many cells qualify (almost everything is default)
comp_strict = discover_complement_branches(features, rare_disease, positive_class=1, center_spec=spec_strict)

# The complement will have higher coverage (more cells are trivially 99% negative)
# This is correct - it's easier to find absence than presence at the margin
```

**Lesson**: Tau interpretation changes perspective:
- Standard @ tau=0.99: "99% of cell members are sick" (rare, hard to find)
- Complement @ tau=0.99: "99% of cell members are healthy" (common, easy to find)

This is expected and appropriate.

---

## Reporting Results from Complement Search

### In a Paper/Report

**Standard language** (adapt as needed):

> We performed exhaustive independent branch discovery to identify feature
> combinations where [positive_class] is concentrated (standard search) and,
> separately, where [positive_class] is absent (complement search). The
> complement search is relevant because the target prevalence is [X]%, making
> direct identification of presence patterns difficult; instead, we identify
> the feature zones where absence is most reliable.
>
> For dimensionality d=2, the complement search identified [K] feature
> combinations, localizing [coverage]% of the non-[positive_class] population
> with pooled purity [purity]%. Cross-validated coverage was [CV]% (SE: [SE]%).
> All centres met a Bonferroni-corrected purity threshold of tau=[tau] at
> family-wise error rate alpha=[alpha].

### In a Dashboard/UI

Add language to distinguish:

| Search Type | Display Label | Legend Text |
|---|---|---|
| Standard | "Presence Pattern" | Green cells: High concentration of [target=value] |
| Complement | "Absence Pattern" | Green cells: High absence of [target=value] |

---

## Debugging Tips

### Q: Complement search returns no branches but standard does

```python
# This can happen! Reasons:
# 1. Complement is easier (majority class), so coverage is often > tau trivially
# 2. Your tau might be too high for the complement
# 3. Your features genuinely don't separate the complement

# Solution: Lower tau or run with n_permutations_centers=999 to see p-values
branches = discover_complement_branches(
    X, Z, positive_class=1,
    center_spec=CenterSpec(tau=0.85),  # Lower threshold
    n_permutations_centers=999,         # See significance
)
```

### Q: Why is complement coverage so high (>90%) but standard is low (<5%)?

```python
# This is CORRECT if your base rate is imbalanced!
# Example: 5% disease, 95% healthy

# Standard: Finds cells with 90% disease (= high concentration of rare class)
# Coverage: 2% (only found a tiny fraction of the rare class)

# Complement: Finds cells with 90% healthy (= high concentration of common class)
# Coverage: 85% (found most of the common class - easier!)

# The two are NOT comparable on the same scale. Each is correct for its purpose.
```

### Q: Do I use the same tau for both?

```python
# Not necessarily. Consider your use case:

# Use case 1: Finding definite diagnosis (standard)
tau_diagnosis = 0.95  # "95% of this cell has disease"

# Use case 2: Finding safe zones (complement)
tau_safe = 0.98  # "98% of this cell are healthy"

# They can differ based on what you're certifying.
```

---

## Next Steps

1. **Try it on your data**: Replace `discover_branches()` with
   `discover_complement_branches()` for high-prevalence targets.

2. **Compare results**: Run both and see which perspective is more useful.

3. **Report carefully**: Clearly state which you used and interpret the coverage
   relative to the non-positive class (not the positive class).

4. **Integrate with frontend** (coming later): Switch between the two modes
   with a radio button or toggle.
