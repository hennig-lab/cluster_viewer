import numpy as np
import scipy.sparse as sp

FIELDS = [
    "chan",
    "cluster_ids",
    "params",
    "spikes",
    "waveform",
    "channel_file_names",
    "n_spikes_excluded",
    "clusters_excluded",
]
FIELDS_TO_IGNORE = ['timeNow']

def compare_struct_fields(s1, s2, fields=FIELDS, tol=0, verbose=True, prefix="", fields_to_ignore=FIELDS_TO_IGNORE):
    """
    Compare specified fields between two MATLAB structs (dicts or np.voids),
    allowing NaNs to match, supporting nested structs and sparse matrices.
    """

    def sparse_equal(a, b, tol=0, path=""):
        """Compare two scipy sparse matrices."""
        if a.shape != b.shape or a.format != b.format:
            if verbose:
                print(f"{path}: sparse matrix shape/format mismatch {a.shape}/{a.format} vs {b.shape}/{b.format}")
            return False
        # Convert to dense only if small; otherwise compare data/indices directly
        if a.nnz != b.nnz:
            if verbose:
                print(f"{path}: sparse matrices have different nnz")
            return False
        if not np.allclose(a.data, b.data, equal_nan=True, atol=tol):
            if verbose:
                print(f"{path}: sparse matrix data differ")
            return False
        if not np.array_equal(a.indices, b.indices) or not np.array_equal(a.indptr, b.indptr):
            if verbose:
                print(f"{path}: sparse matrix indices differ")
            return False
        return True

    def arrays_equal(a, b, tol=0, path=""):
        """Safely compare two numpy arrays, handling nested structs and NaNs."""
        if a.shape != b.shape:
            if verbose:
                print(f"{path}: shape mismatch {a.shape} vs {b.shape}")
            return False

        # Character arrays
        if a.dtype.kind in {'U', 'S'} or b.dtype.kind in {'U', 'S'}:
            eq = np.array_equal(a, b)
            if not eq and verbose:
                print(f"{path}: char arrays differ")
            return eq

        # Boolean arrays
        if a.dtype == bool or b.dtype == bool:
            eq = np.array_equal(a, b)
            if not eq and verbose:
                print(f"{path}: logical arrays differ")
            return eq

        # Arrays of structs
        if a.dtype.names is not None or b.dtype.names is not None:
            for idx in np.ndindex(a.shape):
                if not values_equal(a[idx], b[idx], tol, path=f"{path}[{idx}]"):
                    return False
            return True

        # Object arrays
        if a.dtype == object or b.dtype == object:
            for idx in np.ndindex(a.shape):
                if not values_equal(a[idx], b[idx], tol, path=f"{path}[{idx}]"):
                    return False
            return True

        # Numeric arrays (safe for NaN)
        try:
            eq = np.allclose(a, b, equal_nan=True, atol=tol)
            if not eq and verbose:
                print(f"{path}: numeric arrays differ")
            return eq
        except TypeError:
            # fallback elementwise
            for idx in np.ndindex(a.shape):
                if not values_equal(a[idx], b[idx], tol, path=f"{path}[{idx}]"):
                    return False
            return True

    def values_equal(a, b, tol=0, path=""):
        """Recursive comparison of scalar, array, struct, or sparse objects."""
        if (a is None) != (b is None):
            if verbose:
                print(f"{path}: one is None")
            return False
        if a is None:
            return True

        # Sparse matrices
        if sp.issparse(a) or sp.issparse(b):
            if not (sp.issparse(a) and sp.issparse(b)):
                if verbose:
                    print(f"{path}: one is sparse, other is not")
                return False
            return sparse_equal(a, b, tol, path)

        # Numpy arrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return arrays_equal(a, b, tol, path)

        # MATLAB structs
        if isinstance(a, np.void) or isinstance(b, np.void):
            if not hasattr(a, "dtype") or not hasattr(b, "dtype"):
                if verbose:
                    print(f"{path}: one is not struct-like")
                return False
            fields = a.dtype.names or []
            if fields != (b.dtype.names or []):
                if verbose:
                    print(f"{path}: struct field mismatch")
                return False
            for f in fields:
                if f in fields_to_ignore:
                    continue
                if not values_equal(a[f], b[f], tol, path=f"{path}.{f}"):
                    return False
            return True

        # Scalar numeric values
        if np.isscalar(a) and np.isscalar(b):
            try:
                if np.isnan(a) and np.isnan(b):
                    return True
            except Exception:
                pass
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                if abs(a - b) > tol:
                    if verbose:
                        print(f"{path}: numeric values differ ({a} vs {b})")
                    return False
                return True
            if a != b:
                if verbose:
                    print(f"{path}: scalar values differ ({a} vs {b})")
                return False
            return True

        # Fallback direct comparison
        if a != b:
            if verbose:
                print(f"{path}: values differ ({a} vs {b})")
            return False
        return True

    # Convert np.void to dict if needed
    def to_dict(s):
        if isinstance(s, np.void) and hasattr(s, "dtype") and s.dtype.names:
            return {k: s[k] for k in s.dtype.names}
        return s

    s1d, s2d = to_dict(s1), to_dict(s2)

    # Check that all fields exist
    for f in fields:
        if f not in s1d or f not in s2d:
            if verbose:
                print(f"{prefix}{f}: missing in one struct")
            return False

    # Compare each listed field
    for f in fields:
        if not values_equal(s1d[f], s2d[f], tol, path=prefix + f):
            return False

    return True
