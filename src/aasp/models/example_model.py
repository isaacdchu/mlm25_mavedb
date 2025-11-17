from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model_interface import Model
from .aasp_dataset import AASPDataset

class ExampleModel(Model):
    def __init__(self, params: Dict[str, Any]) -> None:
        pass

    @staticmethod
    def transform(data: pd.DataFrame) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        # data_loader: DataLoader = DataLoader(dataset)
        pass

    def save(self, file_path: str) -> None:
        pass

    @staticmethod
    def load(file_path: str) -> ExampleModel:
        pass
