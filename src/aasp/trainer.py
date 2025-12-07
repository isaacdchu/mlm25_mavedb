from __future__ import annotations
from typing import Dict, Any
from torch.nn import Module
from torch.optim import Optimizer

from .models.model_interface import Model
from .models.aasp_dataset import AASPDataset

class Trainer:
    def __init__(self, model: Model) -> None:
        self.model: Model = model

    def train(
        self,
        train_dataset: AASPDataset,
        test_dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        self.model.train_loop(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            criterion=criterion,
            optimizer=optimizer,
            params=params
        )
