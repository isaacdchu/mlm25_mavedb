# training/src/aasp/data_handler/get_columns.py
from __future__ import annotations
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_columns.py <path-to-pkl>")
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        sys.exit(2)

    import pickle

    # Lazily import optional deps
    def _try(name):
        try:
            return __import__(name)
        except Exception:
            return None

    torch = _try("torch")
    pd = _try("pandas")
    np = _try("numpy")

    obj = None
    load_errors = []

    # ---- Torch with hard MPS→CPU monkeypatch ----
    if torch is not None:
        try:
            import torch.serialization as ts

            def _force_cpu(storage, location):
                # Any serialized location (e.g., 'mps', 'mps:0', etc.) → CPU
                try:
                    return storage.cpu()
                except Exception:
                    return storage  # last resort

            # Monkey-patch the restore function used internally by torch.load
            ts.default_restore_location = _force_cpu  # type: ignore[attr-defined]

            # Use a map_location that also forces CPU
            obj = torch.load(str(p), map_location=_force_cpu, weights_only=False)
        except Exception as e:
            load_errors.append(("torch.load(force_cpu)", e))
            obj = None

    # ---- Try pandas next (may still hit torch reducers) ----
    if obj is None and pd is not None:
        try:
            obj = pd.read_pickle(p)
        except Exception as e:
            load_errors.append(("pandas.read_pickle", e))
            obj = None

    # ---- Raw pickle fallback ----
    if obj is None:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print("Failed to load pickle:", e)
            if load_errors:
                print("Earlier load errors:")
                for where, err in load_errors:
                    print(f" - {where}: {err}")
            sys.exit(3)

    # ---- Extract columns ----
    cols = extract_columns(obj)
    if cols:
        for c in cols:
            print(c)
        sys.exit(0)

    # Diagnostics if we couldn’t find columns
    print("Could not infer column names.")
    print("Top-level object type:", type(obj))
    if isinstance(obj, dict):
        print("Top-level dict keys:", list(obj.keys())[:50])
    elif isinstance(obj, (list, tuple)):
        print("Top-level sequence length:", len(obj))
        print("First 3 element types:", [type(x) for x in obj[:3]])
    sys.exit(4)

def extract_columns(obj):
    try:
        import pandas as pd  # type: ignore
        import numpy as np   # type: ignore
    except Exception:
        pd = None
        np = None

    # DataFrame
    if pd is not None and isinstance(obj, pd.DataFrame):
        return [str(c) for c in obj.columns]

    # Numpy structured array
    if np is not None and isinstance(obj, np.ndarray) and obj.dtype.names:
        return list(obj.dtype.names)

    # Dict containers
    if isinstance(obj, dict):
        for k in ("columns", "feature_names", "feature_names_in_", "cols"):
            if k in obj and isinstance(obj[k], (list, tuple)):
                return [str(x) for x in obj[k]]
        if pd is not None:
            for k in ("df", "dataframe", "X_df"):
                if k in obj and isinstance(obj[k], pd.DataFrame):
                    return [str(c) for c in obj[k].columns]
        if np is not None:
            for k in ("X", "X_train", "X_test"):
                if k in obj and isinstance(obj[k], np.ndarray) and obj[k].dtype.names:
                    return list(obj[k].dtype.names)

    # Lists/tuples
    if isinstance(obj, (list, tuple)):
        if pd is not None:
            for item in obj:
                if isinstance(item, pd.DataFrame):
                    return [str(c) for c in item.columns]
        for item in obj:
            if isinstance(item, (list, tuple)) and all(isinstance(x, (str, int)) for x in item):
                return [str(x) for x in item]

    # sklearn-style
    if hasattr(obj, "feature_names_in_"):
        try:
            return [str(x) for x in obj.feature_names_in_]
        except Exception:
            pass

    return None

if __name__ == "__main__":
    main()
