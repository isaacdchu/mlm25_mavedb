"""
AASPDataset module f
"""

from __future__ import annotations
from typing import Optional, Callable, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch import Tensor

class AASPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        device: str = "cpu",
        transform: Optional[Callable[[pd.DataFrame], None]] = None
    ) -> None:
        self.device: str = device
        if transform:
            transform(data)
        # Convert DataFrame rows to Tensors and store in self.x and self.y
        # Assumes data is a 2D array with numeric features and a "score" column
        self.feature_names: List[str] = [col for col in data.columns if col != "score"]
        x: np.ndarray = data[self.feature_names].to_numpy(dtype=np.float32)  # shape (N, F)
        y: np.ndarray = data["score"].to_numpy(dtype=np.float32)        # shape (N, 1)
        self.x: Tensor = torch.from_numpy(x).to(self.device)
        self.y: Tensor = torch.from_numpy(y).to(self.device)
        self.shape: Tuple[int, ...] = (self.x.shape[0], self.x.shape[1])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (self.x[idx], self.y[idx])

    def __repr__(self) -> str:
        return f"AASPDataset(num_samples={self.shape[0]}, num_features={self.shape[1]})"
