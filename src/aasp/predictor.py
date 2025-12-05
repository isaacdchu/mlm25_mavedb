from __future__ import annotations
from typing import List
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .models.model_interface import Model
from .models.aasp_dataset import AASPDataset

class Predictor:
    def __init__(self, model: Model) -> None:
        self.model: Model = model

    def predict(self, dataset: AASPDataset) -> List[Tensor]:
        data_loader: DataLoader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        predictions: List[Tensor] = []
        self.model.eval()
        (x, y) = next(iter(data_loader))
        x: List[Tensor] = [tensor.to(dtype=torch.float32, device=dataset.device) for tensor in x]
        y: Tensor = y.to(dtype=torch.float32, device=dataset.device)
        output: Tensor = self.model(x)
        predictions.append(output)
        return predictions