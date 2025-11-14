from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
from torch import nn

class ModelInterface(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        input_dim: int,
        cat_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        multi_hot_dims: Optional[Dict[str, int]] = None,
        hidden_dims: Tuple[int, int] = (64, 16),
        dropout_rates: Tuple[float, float] = (0.2, 0.1)
    ):
        pass

    @abstractmethod
    def forward(self, X, **features):
        pass
