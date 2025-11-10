# feature_engineering/data_handler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import pickle
import numpy as np
import yaml

# ----------------------------- Config ---------------------------------

class AASPConfig:
    """
    Reads config YAML and exposes attributes.

    YAML example:
    -------------
    file_path: "data/train/combined_train_data.pkl"

    hyperparameters:
      val_frac: 0.15
      test_frac: 0.0
      seed: 0
    """

    def __init__(self, config_path: str = "config.yaml"):
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r") as f:
            cfg = yaml.safe_load(f) or {}

        self.file_path: str = cfg.get("file_path")

        hp = cfg.get("hyperparameters", {}) or {}
        self.val_frac: float = float(hp.get("val_frac", 0.15))
        self.test_frac: float = float(hp.get("test_frac", 0.0))
        self.seed: int = int(hp.get("seed", 0))

    def __repr__(self) -> str:
        return (f"AASPConfig(file_path={self.file_path!r}, "
                f"val_frac={self.val_frac}, test_frac={self.test_frac}, seed={self.seed})")

# --------------------------- DataHandler ------------------------------

class AASPDataHandler:
    """
    Utility for loading, filtering, encoding, and shaping AASP records.

    - Records are list[dict] from a pickle.
    - Methods here are pure transforms (no PyTorch deps).
    """

    def __init__(self, config: AASPConfig) -> None:
        self.config = config

    # ---- IO ----
    def load_pickle(self, path: Optional[str] = None) -> List[Mapping[str, Any]]:
        """Load raw list-of-dicts from .pkl"""
        p = Path(path or self.config.file_path)
        if not p.exists():
            raise FileNotFoundError(f"Pickle not found: {p}")
        with open(p, "rb") as f:
            obj = pickle.load(f)
        # Normalize to list-of-dicts
        if isinstance(obj, dict):
            # Some pipelines store {"data": [...]}
            obj = obj.get("data", obj.get("records", obj))
        if not isinstance(obj, (list, tuple)):
            raise TypeError(f"Expected list/tuple of records, got {type(obj)}")
        return list(obj)

    # ---- Field selection / filtering ----
    def select_fields(self, records: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]]) \
            -> List[Dict[str, Any]]:
        """
        Keep only input fields + 'score'. Score is *always* preserved.
        """
        if fields is None:
            return [dict(r) for r in records]
        out: List[Dict[str, Any]] = []
        keep = set(fields) | {"score"}
        for r in records:
            out.append({k: r.get(k) for k in keep if k in r})
        return out

    def filter(self,
               records: Sequence[Mapping[str, Any]],
               *,
               predicate: Optional[Callable[[Mapping[str, Any]], bool]] = None,
               scoresets: Optional[Sequence[str]] = None,
               biotypes: Optional[Sequence[str]] = None,
               max_rows: Optional[int] = None) -> List[Mapping[str, Any]]:
        """
        Filter by:
          - predicate(record) -> bool  (custom lambda)
          - scoresets: keep only those scoresets
          - biotypes: keep only those biotypes
          - max_rows: take first N
        """
        scoresets = set(scoresets or [])
        biotypes = set(biotypes or [])
        out: List[Mapping[str, Any]] = []
        for r in records:
            if scoresets and r.get("scoreset") not in scoresets:
                continue
            if biotypes and r.get("biotype") not in biotypes:
                continue
            if predicate and not predicate(r):
                continue
            out.append(r)
            if max_rows is not None and len(out) >= max_rows:
                break
        return out

    # ---- Encoding helpers ----
    def encode(self,
               records: Sequence[Mapping[str, Any]],
               key: str,
               vocab: Mapping[str, int],
               unk_token: str = "<UNK>") -> np.ndarray:
        """
        Map string category -> integer id using vocab. Unknowns -> vocab[unk_token] (or -1 if missing).
        Returns shape [N, 1] int64.
        """
        unk_id = vocab.get(unk_token, -1)
        ids = []
        for r in records:
            v = r.get(key)
            if v is None:
                ids.append(unk_id)
            else:
                ids.append(vocab.get(str(v), unk_id))
        return np.asarray(ids, dtype=np.int64).reshape(-1, 1)

    def one_hot(self, ids: np.ndarray, num_classes: int) -> np.ndarray:
        """
        One-hot encode integer ids. Unknown (<0 or >=num_classes) -> all zeros.
        Input:  [N, 1]  Output: [N, num_classes]
        """
        ids = ids.reshape(-1)
        oh = np.zeros((ids.shape[0], num_classes), dtype=np.float32)
        mask = (ids >= 0) & (ids < num_classes)
        oh[np.arange(ids.shape[0])[mask], ids[mask]] = 1.0
        return oh

    # ---- Targets / embeddings ----
    def get_target(self,
                   records: Sequence[Mapping[str, Any]],
                   key: str = "score",
                   dtype: str = "float32") -> np.ndarray:
        """Extract target vector [N, 1]."""
        y = [r.get(key) for r in records]
        return np.asarray(y, dtype=dtype).reshape(-1, 1)

    def get_embedding(self,
                      records: Sequence[Mapping[str, Any]],
                      key: str,
                      *,
                      pad_to: Optional[int] = None,
                      truncate_to: Optional[int] = None,
                      dtype: str = "float32") -> np.ndarray:
        """
        Build dense [N, D] from per-record embeddings (list/np/torch).
        - If lengths vary, use pad_to and/or truncate_to to force fixed D.
        - If neither provided, infer D from first non-null embedding.
        """
        # Find first non-empty to infer D
        D = None
        for r in records:
            e = r.get(key)
            if e is None:
                continue
            try:
                arr = np.asarray(e, dtype=dtype)
            except Exception:
                # torch Tensor?
                try:
                    import torch  # local import
                    if isinstance(e, torch.Tensor):
                        arr = e.detach().cpu().numpy().astype(dtype, copy=False)
                    else:
                        continue
                except Exception:
                    continue
            D = arr.shape[-1]
            break
        if D is None:
            raise ValueError(f"Could not infer embedding size for key='{key}'")
        if truncate_to is not None:
            D = min(D, int(truncate_to))
        if pad_to is not None:
            D = max(D, int(pad_to))

        out = np.zeros((len(records), D), dtype=dtype)
        for i, r in enumerate(records):
            e = r.get(key)
            if e is None:
                continue
            arr = np.asarray(e, dtype=dtype)
            if truncate_to is not None and arr.shape[-1] > D:
                arr = arr[:D]
            if arr.shape[-1] < D:
                # pad right with zeros
                tmp = np.zeros(D, dtype=dtype)
                tmp[:arr.shape[-1]] = arr
                arr = tmp
            out[i] = arr
        return out

    def fuse_embeddings(self,
                        ref_emb: np.ndarray,
                        alt_emb: np.ndarray,
                        how: str = "concat") -> np.ndarray:
        """
        Combine reference & alternate embeddings -> features.
        """
        if how == "concat":
            return np.concatenate([ref_emb, alt_emb], axis=1)
        if how == "diff":
            return alt_emb - ref_emb
        if how == "sum":
            return alt_emb + ref_emb
        raise ValueError(f"Unknown fuse mode: {how}")

    # ---- Summary ----
    def summary(self, records: Sequence[Mapping[str, Any]]) -> str:
        N = len(records)
        scoresets = {}
        biotypes = {}
        has_keys = set()
        for r in records:
            has_keys.update(k for k, v in r.items() if v is not None)
            ss = r.get("scoreset"); bt = r.get("biotype")
            if ss: scoresets[ss] = scoresets.get(ss, 0) + 1
            if bt: biotypes[bt] = biotypes.get(bt, 0) + 1
        top_ss = sorted(scoresets.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_bt = sorted(biotypes.items(), key=lambda kv: kv[1], reverse=True)[:5]
        msg = [
            f"records: {N}",
            f"keys present: {sorted(has_keys)[:12]}{' ...' if len(has_keys) > 12 else ''}",
            "top scoresets: " + ", ".join(f"{k} (n={v})" for k, v in top_ss) if top_ss else "top scoresets: n/a",
            "top biotypes:  " + ", ".join(f"{k} (n={v})" for k, v in top_bt) if top_bt else "top biotypes: n/a",
        ]
        return "\n".join(msg)
