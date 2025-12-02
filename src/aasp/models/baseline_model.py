from __future__ import annotations
from typing import Dict, Any
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
        self.batch_size: int = abs(params.get('batch_size', 32))
        self.num_epochs: int = abs(params.get('num_epochs', 10))
        class InverseTanh(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return 0.5 * (torch.exp(2 * x) - 1) / (torch.exp(2 * x) + 1)
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=42, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
            InverseTanh()
        )

    @staticmethod
    def transform(data: pd.DataFrame) -> pd.DataFrame:
        data = data[["score", "pos", "ref_long", "alt_long"]]
        data = DataHandler.one_hot_encode(data, columns=["ref_long", "alt_long"])
        return data

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        data_loader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.train()
        self.to(dataset.device)
        for epoch in range(self.num_epochs):
            for batch_idx, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                outputs: Tensor = self.forward(x)
                loss: Tensor = criterion(outputs, y)
                loss.backward()
                optimizer.step()

    def save(self, file_path: str) -> None:
        return

    @staticmethod
    def load(file_path: str) -> BaselineModel:
        return BaselineModel(params={})
