from __future__ import annotations
from typing import Dict, Any
import torch

from .models.model_interface import Model
from .models.aasp_dataset import AASPDataset

class Predictor:
    def __init__(self, model: Model) -> None:
        self.model: Model = model

    def predict(
        self,
        dataset: AASPDataset,
        params: Dict[str, Any]
    ) -> Any:
        pass
