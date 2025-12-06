from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model_interface import Model
from .aasp_dataset import AASPDataset
from .data_handler import DataHandler


class TabularMLPModel(Model):
    """
    Small feedforward MLP over tabular (non-embedding) features.
    Feature Set A: rel_pos + one-hot amino acids + optional biotype/consequence/scoreset.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: Dict[str, Any] = params
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.in_features: int | None = None

    def _ensure_rel_pos(self, data: pd.DataFrame) -> pd.DataFrame:
        if "rel_pos" in data.columns:
            return data
        if "pos" in data.columns:
            # Normalize position by max observed position as a lightweight relative position.
            max_pos: float = float(np.nanmax(data["pos"].astype(float)))
            if max_pos > 0:
                data = data.copy()
                data["rel_pos"] = data["pos"].astype(float) / max_pos
        return data

    def _build_model(self, in_features: int) -> None:
        hidden1: int = int(self.params.get("hidden_dim_1", 128))
        hidden2: int = int(self.params.get("hidden_dim_2", 64))
        dropout: float = float(self.params.get("dropout", 0.0))
        activation: Module = torch.nn.GELU() if self.params.get("use_gelu", True) else torch.nn.ReLU()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden1),
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=hidden1, out_features=hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden2, out_features=1)
        )
        self.in_features = in_features

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = self._ensure_rel_pos(data)
        cat_cols: List[str] = [
            col for col in [
                "ref_long",
                "alt_long",
                "ref_short",
                "alt_short",
                "consequence",
                "biotype",
                "scoreset",
            ]
            if col in data.columns
        ]
        if cat_cols:
            data = DataHandler.one_hot_encode(data, columns=cat_cols)
        drop_cols = [col for col in ["accession", "ensp", "ref_embedding", "alt_embedding", "emb_diff", "sequence", "sequence_length"] if col in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols)
        in_features: int = data.shape[1] - 1  # exclude "score"
        self._build_model(in_features=in_features)
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        if not x:
            raise ValueError("Input feature list is empty.")
        input_tensor: Tensor
        if len(x[0].shape) > 1:
            input_tensor = torch.cat(x, dim=-1)
        else:
            input_tensor = torch.stack(x, dim=0).mT
        return self.model(input_tensor.to(dtype=torch.float32))

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        batch_size: int = abs(params.get("batch_size", 32))
        num_epochs: int = abs(params.get("num_epochs", 10))
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train()
        self.to(device=dataset.device)
        for _ in range(num_epochs):
            for x, y in data_loader:
                optimizer.zero_grad()
                features: List[Tensor] = [tensor.to(device=dataset.device, dtype=torch.float32) for tensor in x]
                targets: Tensor = y.to(dtype=torch.float32, device=dataset.device)
                outputs: Tensor = self.forward(features)
                loss: Tensor = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def save(self, file_path: str) -> None:
        state = {
            "state_dict": self.state_dict(),
            "params": self.params,
            "in_features": self.in_features,
        }
        torch.save(state, file_path)

    @staticmethod
    def load(file_path: str) -> TabularMLPModel:
        checkpoint = torch.load(file_path, map_location="cpu")
        model = TabularMLPModel(params=checkpoint.get("params", {}))
        in_features = checkpoint.get("in_features")
        if in_features is None:
            raise ValueError("Missing in_features in checkpoint; cannot reconstruct model.")
        model._build_model(in_features=in_features)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
