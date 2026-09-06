# Git Commit Template for Complement Branch Discovery

Use this as a template for your git commit when integrating these changes:

```
Implement Complement Branch Discovery - find feature combinations where target is absent

Summary
-------

Add discover_complement_branches() function for identifying feature combinations
where the positive_class is ABSENT (high concentration in complement).

This extension is critical for high-base-rate targets (e.g., 90%+ positive class)
where standard branch discovery is uninformative. The complement search inverts
the binary indicator (z_complement = 1 - z_binary) and applies all existing
metrics unchanged, providing independent statistical guarantees.

Use Cases
---------
- Medical: Find cells where disease is ABSENT (not present)
- Marketing: Identify guaranteed non-responders for exclusion
- Quality: Locate defect-free process parameter zones
- Fraud: Find high-trust transaction regions (0.1% fraud)
- Public Health: Identify disease-free demographic pockets

Design
------
- Single line change: invert binary indicator before exhaustive search
- Zero code duplication: reuses entire pipeline (factory, search, reporting)
- Identical signatures and statistical guarantees as discover_branches()
- Backward compatible: no changes to existing code

Files Changed
-------------
vsf/__init__.py
  - Import discover_complement_branches from complement module
  - Export in __all__

New Files
---------
vsf/complement.py (~200 lines)
  - discover_complement_branches() function
  - Full docstring with examples and mathematical foundation
  - Production-ready with comprehensive type hints

tests/test_complement.py (~400 lines)
  - 7 test classes, 15+ tests
  - Covers: basics, statistics, null data, synergy, edge cases
  - Ensures mathematical correctness and consistency

Testing
-------
All tests pass:
  pytest tests/test_complement.py -v
  pytest tests/ -v (no regressions in existing tests)

Documentation
--------------
See INTEGRATION_GUIDE.md and EXAMPLES_AND_USE_CASES.md for:
- Step-by-step integration instructions
- Architecture diagram
- Design decisions
- Real-world use case examples
- Edge case analysis and debugging tips
- Frontend integration notes (deferred)

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
```

## How to Use

1. Stage your changes:
```bash
git add vsf/complement.py vsf/__init__.py tests/test_complement.py
```

2. Copy the message above and use it for your commit:
```bash
git commit
# Paste the message above in your editor
```

Or use it directly:
```bash
git commit -m "Implement Complement Branch Discovery - find feature combinations where target is absent" \
           -m "$(cat commit_body.txt)"
```

3. After commit, verify:
```bash
git log -1 --stat
git show
```

## Alternative: Shorter Commit Message

If you prefer a more concise commit:

```
Add discover_complement_branches() for high-base-rate targets

Enables searching for feature combinations where positive_class is absent.
Critical for targets with 80%+ prevalence where standard search is uninformative.

- discover_complement_branches(): ~200 lines, identical signature to discover_branches()
- Inverts binary indicator only (z_complement = 1 - z_binary)
- Reuses entire existing pipeline - zero duplication
- Comprehensive test suite (15+ tests, 7 test classes)
- Production-ready with full statistical guarantees

See INTEGRATION_GUIDE.md and EXAMPLES_AND_USE_CASES.md for details.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
```

## Version Bump

Update `__version__` in `vsf/__init__.py`:

**From:**
```python
__version__ = "2.3.0"
```

**To:**
```python
__version__ = "2.4.0"
```

This reflects the addition of a new major feature (complement search).
