from __future__ import annotations
from typing import Dict, Any
from abc import ABC, abstractmethod
import pandas as pd
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .aasp_dataset import AASPDataset

class Model(Module, ABC):
    @abstractmethod
    def __init__(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError("Model is an abstract class and cannot be instantiated directly.")

    @staticmethod
    @abstractmethod
    def transform(data: pd.DataFrame) -> None:
        raise NotImplementedError("Subclasses must implement the transform method.")

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        raise NotImplementedError("Subclasses must implement the train_loop method.")

    @abstractmethod
    def save(self, file_path: str):
        raise NotImplementedError("Subclasses must implement the save method.")

    @staticmethod
    @abstractmethod
    def load(file_path: str) -> Model:
        raise NotImplementedError("Subclasses must implement the load method.")
