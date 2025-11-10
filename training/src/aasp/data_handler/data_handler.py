"""
Class for handling data operations in the AASP module.
Loads training/test data, manages data splits, and provides data loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


@dataclass
class AASPConfig:
    """
    Minimal config object. Extend later as needed.
    """
    pkl_path: str
    fields: List[str]
    val_frac: float = 0.15
    test_frac: float = 0.0
    seed: int = 42
    group_by: Optional[str] = None


class AASPDataHandler:
    """
    Concrete class – NOT an interface.

    Methods are empty stubs right now but *this object exists*.
    The AASPDataset can already depend on the shape / names of these APIs.
    """

    def __init__(self, config: AASPConfig):
        """
        Just store config, do not load anything here
        """
        self.config = config

    # -------------------------------------------------------------------------
    def load_pickle(self, path: Optional[str] = None) -> List[Mapping[str, Any]]:
        """load the raw list of dict records from .pkl"""
        raise NotImplementedError

    def select_fields(self, records, fields):
        """return new records list with only selected keys"""
        raise NotImplementedError

    def filter(self, records, *, scoresets=None, biotypes=None, max_rows=None):
        """return filtered list of records"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def split(self, records, *, val_frac=None, test_frac=None, seed=None, group_by=None):
        """return (train_records, val_records, test_records)"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def fit_vocab(self, records, key: str, add_unk=True, unk_token="<UNK>"):
        """build category→id mapping"""
        raise NotImplementedError

    def encode(self, records, key: str, vocab: Mapping[str, int], unk_token="<UNK>"):
        """convert string categories → integer ids"""
        raise NotImplementedError

    def one_hot(self, ids: np.ndarray, num_classes: int) -> np.ndarray:
        """convert integer ids to one-hot matrix"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def get_numeric(self, records, keys, dtype="float32") -> np.ndarray:
        """extract numeric columns into 2D float array"""
        raise NotImplementedError

    def get_target(self, records, key="score", dtype="float32") -> np.ndarray:
        """extract target score column"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def get_embedding(self, records, key, *, pad_to=None, truncate_to=None, dtype="float32"):
        """extract stored embedding vectors"""
        raise NotImplementedError

    def fuse_embeddings(self, ref_emb, alt_emb, how="concat"):
        """combine ref + alt embeddings into 1 vector"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def get_sequence(self, records, key="sequence") -> List[str]:
        """return list of raw sequence strings"""
        raise NotImplementedError

    def tokenize_sequence(self, seqs, alphabet=tuple("ACDEFGHIKLMNPQRSTVWY"), unk_token="X"):
        """simple mapping: AA char → integer"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def fit_scaler(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """compute mean/std stats"""
        raise NotImplementedError

    def apply_scaler(self, X, stats):
        """standardize using stats"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def cache_arrays(self, name: str, **arrays):
        """write npz of arrays"""
        raise NotImplementedError

    def load_cached(self, name: str) -> Dict[str, np.ndarray]:
        """load npz arrays"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def summary(self, records):
        """pretty text summary"""
        raise NotImplementedError
