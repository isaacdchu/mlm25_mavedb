from __future__ import annotations
from typing import Optional, Callable, List, Tuple
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor

class AASPDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        device: str = "cpu",
        transform: Optional[Callable[[pd.DataFrame], None]] = None
    ) -> None:
        self.x: List[Tensor] = []
        self.y: List[Tensor] = []
        self.device: str = device
        if transform:
            transform(data)
        # Convert DataFrame rows to Tensors and store in self.x and self.y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (self.x[idx], self.y[idx])
