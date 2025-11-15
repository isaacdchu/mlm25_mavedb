from __future__ import annotations
from typing import List
from torch import Tensor

from .models.model_interface import Model
from .models.aasp_dataset import AASPDataset

class Predictor:
    def __init__(self, model: Model) -> None:
        self.model: Model = model

    def predict(self, dataset: AASPDataset) -> List[Tensor]:
        pass
