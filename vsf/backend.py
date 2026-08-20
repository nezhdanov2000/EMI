"""
VSF GPU Backend Module
Dynamically loads CuPy if available and a CUDA GPU is present, otherwise falls back to NumPy.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

import importlib

# Attempt to dynamically import cupy and verify CUDA device & runtime libs
try:
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() > 0:
        # Verify basic memory allocation and random generation work
        _test = cp.zeros(2, dtype=cp.int32)
        _ = cp.random.permutation(2)
        HAS_GPU = True
    else:
        HAS_GPU = False
        cp = None
except Exception as exc:
    logger.debug("CuPy GPU backend initialization failed: %s. Using NumPy fallback.", exc)
    cp = None
    HAS_GPU = False

# Global state for backend
_use_gpu = HAS_GPU

def set_backend(use_gpu: bool):
    """Force enable or disable GPU backend (if available)."""
    global _use_gpu
    if use_gpu and not HAS_GPU:
        logger.warning("CuPy or CUDA GPU is not available. Falling back to NumPy.")
        _use_gpu = False
    else:
        _use_gpu = use_gpu

def get_backend():
    """Returns the active backend (cupy or numpy)."""
    if _use_gpu and cp is not None:
        return cp
    return np

def as_numpy(arr):
    """Converts a CuPy array or generic array to NumPy ndarray."""
    if HAS_GPU and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def as_backend(arr):
    """Converts a NumPy or CuPy array to the active backend (CuPy or NumPy)."""
    xp = get_backend()
    if xp is not np and cp is not None:
        if isinstance(arr, cp.ndarray):
            return arr
        arr_np = np.asarray(arr)
        # CuPy cannot handle string/object arrays, so encode them into integer indices
        if arr_np.dtype.kind in ('U', 'S', 'O', 'b'):
            if arr_np.ndim > 1:
                cols = []
                for j in range(arr_np.shape[1]):
                    _, col_idx = np.unique(arr_np[:, j], return_inverse=True)
                    cols.append(col_idx)
                arr_np = np.column_stack(cols)
            else:
                _, arr_np = np.unique(arr_np, return_inverse=True)
        return xp.asarray(arr_np)
    elif HAS_GPU and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
