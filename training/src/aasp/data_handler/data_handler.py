"""
Class for handling data operations in the AASP module.
Loads training/test data, manages data splits, and provides data loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Callable, Sequence
import numpy as np
import yaml
from pathlib import Path
import pickle
import pandas as pd

class AASPConfig:
    """
    Loads configuration parameters from config.yaml and exposes them as attributes.

    Example YAML shape:
    -------------------
    file_path: "../../../data/train/combined_train_data.pkl"
    hyperparameters:
      learning_rate: 0.0001
      val_frac:      0.15
      test_frac:     0.0
      seed:          0
    """

    def __init__(self, config_path: str = "config.yaml"):
        self._path = Path(config_path)

        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")

        with open(self._path, "r") as f:
            cfg = yaml.safe_load(f)

        # top-level
        self.file_path     = cfg.get("file_path")

        # hyperparams block
        param              = cfg.get("hyperparameters", {})
        self.val_frac      = float(param.get("val_frac", 0.15))
        self.test_frac     = float(param.get("test_frac", 0.0))
        self.seed          = int(param.get("seed", 0))

    def __repr__(self):
        return (
            f"AASPConfig(file_path={self.file_path}, "
            f"lr={self.learning_rate}, "
            f"val_frac={self.val_frac}, test_frac={self.test_frac}, seed={self.seed})"
        )


class AASPDataHandler:
    """
    Concrete class – NOT an interface.

    Methods are empty stubs right now but *this object exists*.
    The AASPDataset can already depend on the shape / names of these APIs.
    """

    def __init__(self, config: AASPConfig) -> None:
        """
        Just store config, do not load anything here
        """
        self.config: AASPConfig = config

    # -------------------------------------------------------------------------
    def load_pickle(self, path: Optional[str] = None) -> List[Mapping[str, Any]]:
        """load the raw list of dict records from .pkl"""
        # 1) decide which path to use
        src = path if path is not None else self.config.file_path

        # 2) resolve relative to the config YAML location
        # e.g., config.yaml in .../project/config/config.yaml
        # and file_path: ../../data/train/combined_train_data.pkl
        cfg_dir = self.config._path.parent
        p = (cfg_dir / src).resolve() if not Path(src).is_absolute() else Path(src)

        if not p.exists():
            raise FileNotFoundError(f"Pickle file not found: {p}")

        # 3) load the pickle
        with open(p, "rb") as f:
            obj = pickle.load(f)

        # 4) normalize to list-of-dicts
        #    - case A: already list-of-dicts
        if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], Mapping)):
            return obj  # type: ignore[return-value]

        #    - case B: pandas DataFrame (lazy import)
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")  # type: ignore[return-value]
        except Exception:
            # If pandas isn't installed, we'll fall through to error below.
            pass

        #    - otherwise we don't know how to handle it
        raise TypeError(
            "Unsupported pickle format. Expected a list[dict] or a pandas DataFrame."
        )

    def select_fields(
        self,
        records: Sequence[Mapping[str, Any]],
        fields: Optional[Sequence[str]] = None,
    ) -> List[Mapping[str, Any]]:
        """
        Return new records where only the requested *input feature* keys are kept,
        while always preserving the target 'score' (if present).

        Parameters
        ----------
        records : Sequence[Mapping[str, Any]]
            Input records (list of dict-like items).
        fields : Optional[Sequence[str]]
            Names of input features to keep. If None, uses self.config.fields.

        Returns
        -------
        List[Mapping[str, Any]]
            New list of dicts with:
            - requested input fields (missing values filled with None)
            - plus 'score' if it exists in the original record.

        Notes
        -----
        - We keep 'score' even though it isn't part of `fields` by design.
        - We fill missing requested fields with None so downstream array-building
        can assume a stable set of keys.
        """
        if fields is None:
            if not getattr(self.config, "fields", None):
                raise ValueError(
                    "select_fields: no fields provided and self.config.fields is empty."
                )
            fields = self.config.fields

        wanted = list(fields)  # ensure list
        out: List[Dict[str, Any]] = []

        for rec in records:
            # Build a new shallow dict with only desired inputs
            row: Dict[str, Any] = {k: rec.get(k, None) for k in wanted}

            # Always keep target if present
            if "score" in rec:
                row["score"] = rec["score"]

            out.append(row)

        return out

    # Change it to take a lamda or function
    # Usage ex: filtered = handler.filter(records, lambda r: r["scoreset"] == "urn:mavedb:00000069-a-2")
    def filter(
        self,
        records: Sequence[Mapping[str, Any]],
        predicate: Callable[[Mapping[str, Any]], bool]
    ) -> List[Mapping[str, Any]]:
        """
        Apply a filtering predicate to the records.

        Parameters
        ----------
        records : Sequence[Mapping[str, Any]]
            The input records (typically list of dicts).
        predicate : Callable[[Mapping[str, Any]], bool]
            A function/lambda that takes a single record and returns True if it should remain.

        Returns
        -------
        List[Mapping[str, Any]]
            A new list containing only the records for which predicate(record) is True.

        Example
        -------
        handler.filter(records, lambda r: r["scoreset"] == "some_id")
        """
        if not callable(predicate):
            raise TypeError("filter requires a callable (e.g., lambda r: condition )")

        return [rec for rec in records if predicate(rec)]
    
    def fit_vocab(
        self,
        records: Sequence[Mapping[str, Any]],
        key: str,
        add_unk: bool = True,
        unk_token: str = "<UNK>",
    ) -> Dict[str, int]:
        """
        Build a deterministic mapping from category string → integer ID for a given field.

        Parameters
        ----------
        records : Sequence[Mapping[str, Any]]
            Source records (use TRAIN ONLY to avoid leakage).
        key : str
            Field name to index (e.g., "scoreset", "ref_long", "alt_long", "biotype").
        add_unk : bool, default True
            Include a special unknown-token for unseen/missing categories.
        unk_token : str, default "<UNK>"
            Label for the unknown-token entry in the vocab.

        Returns
        -------
        Dict[str, int]
            Mapping category → ID. If `add_unk` is True, `<UNK>` is ID 0 and
            real categories start from 1. Otherwise, categories start from 0.

        Notes
        -----
        - Only string-able values are included. None/missing values are ignored here
        (they will map to <UNK> during encode()).
        - Categories are sorted to keep IDs stable across runs.
        """
        # Collect unique, non-missing values
        seen: set[str] = set()
        for rec in records:
            val = rec.get(key, None)
            if val is None:
                continue
            # Normalize to string for safety
            seen.add(str(val))

        # Deterministic order
        cats: List[str] = sorted(seen)

        vocab: Dict[str, int] = {}
        next_id = 0

        if add_unk:
            vocab[unk_token] = next_id
            next_id += 1

        for c in cats:
            vocab[c] = next_id
            next_id += 1

        return vocab


    # -------------------------------------------------------------------------
    def encode(
        self,
        records: Sequence[Mapping[str, Any]],
        key: str,
        vocab: Mapping[str, int],
        unk_token: str = "<UNK>",
    ) -> np.ndarray:
        """
        Convert a categorical field into integer IDs using a provided vocabulary.

        Parameters
        ----------
        records : Sequence[Mapping[str, Any]]
            List-like of dict records (one per variant row).
        key : str
            Field to encode (e.g., "scoreset", "ref_long", "alt_long").
        vocab : Mapping[str, int]
            Prebuilt mapping from category string → integer ID (from fit_vocab on TRAIN).
        unk_token : str, default "<UNK>"
            Token in `vocab` used for missing/unseen categories.

        Returns
        -------
        np.ndarray (int64) of shape [N]
            Integer IDs aligned with the input `records` order.

        Behavior
        --------
        - If the record lacks `key` or its value is not in `vocab`, the ID for
        `unk_token` is used.
        - Values are normalized to str before lookup for robustness.
        """
        # Ensure vocab has an <UNK> entry
        if unk_token not in vocab:
            raise ValueError(
                f"encode: vocab has no entry for unk_token '{unk_token}'. "
                "Build vocab with add_unk=True in fit_vocab."
            )
        unk_id = vocab[unk_token]

        ids: List[int] = []
        for rec in records:
            val = rec.get(key, None)
            # Normalize to string for lookup; None → <UNK>
            if val is None:
                ids.append(unk_id)
            else:
                sval = str(val)
                ids.append(vocab.get(sval, unk_id))

        return np.asarray(ids, dtype=np.int64)

    def one_hot(self, ids: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert integer IDs to a 1-of-K one-hot matrix.

        Parameters
        ----------
        ids : np.ndarray
            Array of integer class IDs of shape [N]. Must satisfy 0 <= ids[i] < num_classes.
        num_classes : int
            Number of classes (one-hot width).

        Returns
        -------
        np.ndarray of shape [N, num_classes], dtype float32
            One-hot matrix where row i has a 1.0 at column ids[i] and 0.0 elsewhere.

        Notes
        -----
        - This is mainly for baselines/sanity checks. For large-cardinality categories,
        prefer learned embeddings in the model.
        """
        if ids.ndim != 1:
            raise ValueError(f"one_hot: expected a 1D array, got shape {ids.shape}")
        if num_classes <= 0:
            raise ValueError(f"one_hot: num_classes must be > 0, got {num_classes}")

        # Ensure integer type
        ids = ids.astype(np.int64, copy=False)

        # Validate range explicitly (fail fast helps catch leaking/encoding bugs)
        min_id = ids.min(initial=0)
        max_id = ids.max(initial=-1)
        if min_id < 0 or max_id >= num_classes:
            raise ValueError(
                f"one_hot: id out of range (min={min_id}, max={max_id}) for num_classes={num_classes}. "
                "Check your vocab/encode logic."
            )

        # Eye-indexing is simple and fast for moderate sizes
        oh = np.eye(num_classes, dtype=np.float32)[ids]
        return oh

    # -------------------------------------------------------------------------
    def get_target(
        self,
        records: Sequence[Mapping[str, Any]],
        key: str = "score",
        dtype: str = "float32"
    ) -> np.ndarray:
        """
        Extract the target score column into a numpy array.

        Parameters
        ----------
        records : Sequence[Mapping[str, Any]]
            List-like of dicts.
        key : str, default "score"
            Field to use for the target.
        dtype : str, default "float32"
            Numpy dtype for the returned array.

        Returns
        -------
        np.ndarray of shape [N, 1]
            Column vector of target values aligned with input order.
        """
        values = []
        for rec in records:
            val = rec.get(key, None)
            if val is None:
                raise KeyError(f"get_target: record missing required key '{key}'")
            values.append(val)

        arr = np.asarray(values, dtype=dtype).reshape(-1, 1)
        return arr

    # -------------------------------------------------------------------------
    def summary(self, records) -> str:
        """
        Produce a compact text summary of the records for quick inspection.
        This does not compute statistics — it's just for sanity checking.

        Returns
        -------
        str
            Formatted multi-line string.
        """
        n = len(records)
        if n == 0:
            return "summary: no records"

        # sample record for key structure inspection
        sample = records[0]

        lines = []
        lines.append(f"Total records: {n}")
        lines.append(f"Keys present: {list(sample.keys())}")

        # print example row (first one)
        lines.append(f"Sample record: {sample}")

        # optionally show top 5 common values for common categorical features
        for field in ["scoreset", "ref_long", "alt_long", "biotype"]:
            # only show if key exists
            if field in sample:
                counts = {}
                for r in records:
                    val = r.get(field, None)
                    counts[val] = counts.get(val, 0) + 1
                # sort by frequency descending
                top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                lines.append(f"Top {field} values: {top5}")

        return "\n".join(lines)