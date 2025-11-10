"""
Class for handling data operations in the AASP module.
Loads training/test data, manages data splits, and provides data loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import yaml
from pathlib import Path

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
        raise NotImplementedError

    def select_fields(self, records, fields):
        """return new records list with only selected keys"""
        raise NotImplementedError

    # Change it to take a lamda or function
    def filter():
        """return filtered list of records"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def encode(self, records, key: str, vocab: Mapping[str, int], unk_token="<UNK>"):
        """convert string categories → integer ids"""
        raise NotImplementedError

    def one_hot(self, ids: np.ndarray, num_classes: int) -> np.ndarray:
        """convert integer ids to one-hot matrix"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def get_target(self, records, key="score", dtype="float32") -> np.ndarray:
        """extract target score column"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def summary(self, records):
        """pretty text summary"""
        raise NotImplementedError
