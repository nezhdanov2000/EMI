import numpy as np
import pandas as pd
import pytest
from vsf.mining import mine_dirty_center, nmi_miller_madow_corrected

def test_nmi_miller_madow_corrected():
    # Test perfect correlation
    Z = np.array([0, 0, 0, 1, 1, 1])
    phi = np.array([0, 0, 0, 1, 1, 1])
    nmi = nmi_miller_madow_corrected(Z, phi)
    assert 0.0 <= nmi <= 1.0

    # Test small sample
    Z_small = np.array([1])
    phi_small = np.array([1])
    assert nmi_miller_madow_corrected(Z_small, phi_small) == 0.0

def test_mine_dirty_center_synthetic():
    df = pd.DataFrame({
        "f1": ["a", "a", "a", "a", "b", "b", "b", "b"] * 10,
        "f2": ["x", "x", "y", "y", "x", "x", "y", "y"] * 10,
        "f3": ["m", "n", "m", "n", "m", "n", "m", "n"] * 10,
    })
    # Target Z is 1 if f1 == 'a' and f2 == 'x'
    Z = ((df["f1"] == "a") & (df["f2"] == "x")).astype(int).values
    
    # Dirty center: select all samples
    mask = np.ones(len(df), dtype=bool)
    
    results = mine_dirty_center(df, Z, mask, I_Z_X_F=1.0, max_depth=2, top_t1=5)
    
    assert len(results) > 0
    best = results[0]
    assert "conditions" in best
    assert "human_text" in best
    assert "purity_pos" in best
    assert "nmi_local" in best
    assert best["nmi_local"] > 0.0
