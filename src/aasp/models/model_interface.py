"""
Model interface for AASP models
Defines the abstract base class that all AASP models must implement
Wrapper around torch.nn.Module
"""

from __future__ import annotations
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .aasp_dataset import AASPDataset

class Model(Module, ABC):
    """
    Abstract base class for AASP models
    """

    def __init__(self) -> None:
        """
        Should initialize the model with given parameters
        Args:
            params (Dict[str, Any]):
                Model-specific parameters
        Returns: None
        """
        super().__init__()

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Data processing method specific to the model
        Used in conjunction with AASPDataset
        Args:
            data (pd.DataFrame): Input data to transform
        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the transform method.")

    @abstractmethod
    def forward(self, x: List[Tensor]) -> Tensor:
        """
        Forward pass of the model
        Args:
            x (Tensor): Input tensor
        Returns: Tensor
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        """
        Training loop for the model
        Args:
            dataset (AASPDataset): Dataset for training
            criterion (Module): Loss function
            optimizer (Optimizer): Optimizer for training
            params (Dict[str, Any]): Additional training parameters
        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the train_loop method.")

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Save the model to a file
        Args:
            file_path (str): Path to save the model
        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the save method.")

    @staticmethod
    @abstractmethod
    def load(file_path: str) -> Model:
        """
        Load a model from a file
        Args:
            file_path (str): Path to load the model from
        Returns: Model
            Loaded model instance
        """
        raise NotImplementedError("Subclasses must implement the load method.")
