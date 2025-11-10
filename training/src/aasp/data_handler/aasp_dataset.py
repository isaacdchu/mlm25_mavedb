# feature_engineering/aasp_dataset.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from training.src.aasp.data_handler.data_handler import AASPConfig, AASPDataHandler

class AASPDataset(Dataset):
    """
    PyTorch Dataset for AASP.
    Expects a pickle with list-of-dicts records containing:
      - ref_embedding, alt_embedding (vectors)
      - score (float)
      - optional categorical fields (scoreset, biotype, consequence, ...)
    """

    def __init__(self,
                 config_path: str = "config.yaml",
                 fields: Optional[Sequence[str]] = None,
                 fuse_mode: str = "concat",
                 pad_to: Optional[int] = None,
                 truncate_to: Optional[int] = None):
        super().__init__()
        self.cfg = AASPConfig(config_path)
        self.handler = AASPDataHandler(self.cfg)

        # Load raw data
        self.records = self.handler.load_pickle(self.cfg.file_path)

        # Optional narrow to fields (inputs only; score is always kept)
        self.records = self.handler.select_fields(self.records, fields)

        # Build X (features) and y (target) now (eager), or lazily in __getitem__.
        # We'll do eager → faster training epoch over epoch.
        ref = self.handler.get_embedding(self.records, "ref_embedding",
                                         pad_to=pad_to, truncate_to=truncate_to)
        alt = self.handler.get_embedding(self.records, "alt_embedding",
                                         pad_to=pad_to, truncate_to=truncate_to)
        X = self.handler.fuse_embeddings(ref, alt, how=fuse_mode)  # [N, D]
        y = self.handler.get_target(self.records, "score")          # [N, 1]

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
