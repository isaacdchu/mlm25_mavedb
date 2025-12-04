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
        x (List[Tuple[Tensor, ...]]):
            List of length N of tuples of length F of feature tensors.
            Each feature tensor has shape (E,) where
                N = number of samples
                F = number of features
                E = embedding size (1 for scalar features)
        y (List[Tensor]):
            List of all label tensor (N, 1) where N is number of samples
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
        
        # Initialize self.x and self.y
        # Extract labels (score column) and convert to a list of tensors
        self.y: List[Tensor] = [
            torch.tensor([label], device=self.device, dtype=torch.float32)
            for label in data["score"].values
        ]

        # Extract features (all columns except "score") and convert to a list of lists of tensors
        self.x: List[List[Tensor]] = [
            [
                torch.tensor(value, device=self.device, dtype=torch.float32)
                if np.isscalar(value)
                else torch.tensor(value, device=self.device, dtype=torch.float32)
                for value in row
            ]
            for row in data.drop(columns=["score"]).values
        ]

        # Store the shape of the dataset
        self.shape: Tuple[int, int] = (len(self.y), len(self.x[0]) if self.x else 0)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset
        Returns: int
            Number of samples
        """
        return self.shape[0]

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], Tensor]:
        """
        Get the feature tensor and label tensor for a given index
        Args:
            idx (int):
                Index of the sample to retrieve
        Returns: Tuple[List[Tensor], Tensor]
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
