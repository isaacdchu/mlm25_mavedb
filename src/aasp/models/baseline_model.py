from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model_interface import Model
from .aasp_dataset import AASPDataset
from .data_handler import DataHandler

class BaselineModel(Model):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.model: torch.nn.Sequential = torch.nn.Sequential() # to be defined in transform
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[["score", "pos", "ref_long", "alt_long"]]
        data = DataHandler.one_hot_encode(data, columns=["ref_long", "alt_long"])
        in_features: int = data.shape[1] - 1 # Exclude the "score" column
        class InverseTanh(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return 0.5 * (torch.exp(2 * x) - 1) / (torch.exp(2 * x) + 1)
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=in_features, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
            InverseTanh()
        )
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        input_tensor: Tensor = torch.cat(x, dim=-1) if len(x[0].shape) > 1 else torch.stack(x, dim=0)
        return self.model(input_tensor.mT)

    def train_loop(
        self,
        train_dataset: AASPDataset,
        test_dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        batch_size: int = abs(params.get('batch_size', 32))
        num_epochs: int = abs(params.get('num_epochs', 10))
        if train_dataset.device != test_dataset.device:
            raise ValueError("Train and test datasets must be on the same device")
        device = train_dataset.device
        data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.train()
        self.to(device=device)
        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                x: List[Tensor] = [tensor.to(dtype=torch.float32, device=device) for tensor in x]
                y: Tensor = y.to(dtype=torch.float32, device=device)
                outputs: Tensor = self.forward(x)
                loss: Tensor = criterion(outputs, y)
                loss.backward()
                optimizer.step()

    def save(self, file_path: str) -> None:
        return

    @staticmethod
    def load(file_path: str) -> BaselineModel:
        return BaselineModel(params={})
