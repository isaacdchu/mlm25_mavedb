"""
AASPDataset class for PyTorch
"""

from __future__ import annotations
from typing import Optional, Callable, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

class AASPDataset(Dataset):
    """
    PyTorch Dataset for loading data from a pandas DataFrame into PyTorch Tensors
    Members variables:
        x (Tensor):
            Feature tensor (N, F) where N is number of samples and F is number of features
        y (Tensor):
            Label tensor (N, 1) where N is number of samples
        shape (Tuple[int, ...]):
            Shape of the dataset (N, F)
        transform (Optional[Callable[[pd.DataFrame], None]]):
            Optional transformation function applied to the DataFrame
        device (str):
            Device to store the tensors on (eg "cpu" or "cuda")
        feature_names (List[str]):
            List of feature column names in the DataFrame
    """

    def __init__(
        self,
        data: pd.DataFrame,
        device: str = "cpu",
        transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    ) -> None:
        """
        Initialize the AASPDataset with a pandas DataFrame
        Assumes that the DataFrame contains a "score" column for labels,
        and all other columns are numeric scalar features
        Args:
            data (pd.DataFrame):
                Input data with features and "score" column
            device (str):
                Device to store the tensors on ("cpu" or "cuda")
            transform (Optional[Callable[[pd.DataFrame], None]]):
                Optional transformation function to apply to the DataFrame
        Returns: None
            None
        """
        self.device: str = device
        self.transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = transform
        if transform:
            data = transform(data)
        # Convert DataFrame rows to Tensors and store in self.x and self.y
        # Assumes data is a 2D array with numeric features and a "score" column
        self.feature_names: List[str] = [col for col in data.columns if col != "score"]
        x: np.ndarray = data[self.feature_names].to_numpy(dtype=np.float32)  # shape (N, F)
        y: np.ndarray = data["score"].to_numpy(dtype=np.float32)        # shape (N, 1)
        self.x: Tensor = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        self.y: Tensor = torch.from_numpy(y).to(device=self.device, dtype=torch.float32).unsqueeze(1)
        self.shape: Tuple[int, ...] = (self.x.shape[0], self.x.shape[1])

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset
        Returns: int
            Number of samples
        """
        return self.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the feature tensor and label tensor for a given index
        Args:
            idx (int):
                Index of the sample to retrieve
        Returns: Tuple[Tensor, Tensor]
            Tuple containing the feature tensor and label tensor
        """
        return (self.x[idx], self.y[idx])

    def __repr__(self) -> str:
        """
        Return a string representation of the dataset
        Returns: str
            String representation
        """
        return f"AASPDataset(num_samples={self.shape[0]}, num_features={self.shape[1]})"
