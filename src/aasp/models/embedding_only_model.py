from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model_interface import Model
from .aasp_dataset import AASPDataset


class EmbeddingOnlyModel(Model):
    """
    MLP operating purely on ESM embedding differences (no tabular features).
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: Dict[str, Any] = params
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.input_dim: int | None = None

    def _build_model(self, input_dim: int) -> None:
        hidden: int = int(self.params.get("hidden_dim", 256))
        hidden2: int = int(self.params.get("hidden_dim_2", 128))
        dropout: float = float(self.params.get("dropout", 0.1))
        activation = torch.nn.GELU() if self.params.get("use_gelu", True) else torch.nn.ReLU()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden),
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=hidden, out_features=hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden2, out_features=1)
        )
        self.input_dim = input_dim

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if "ref_embedding" not in data.columns or "alt_embedding" not in data.columns:
            raise KeyError("Both 'ref_embedding' and 'alt_embedding' columns are required for EmbeddingOnlyModel.")
        data = data[["score", "ref_embedding", "alt_embedding"]].copy()
        def _compute_diff(row: pd.Series) -> Tensor:
            ref: Tensor = torch.as_tensor(row["ref_embedding"], dtype=torch.float32)
            alt: Tensor = torch.as_tensor(row["alt_embedding"], dtype=torch.float32)
            return torch.nan_to_num(alt - ref)
        data["emb_diff"] = data.apply(_compute_diff, axis=1)
        data = data.drop(columns=["ref_embedding", "alt_embedding"])
        emb_dim: int = len(data["emb_diff"].iloc[0])
        self._build_model(input_dim=emb_dim)
        data = data[["score", "emb_diff"]]
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        if not x:
            raise ValueError("Input feature list is empty.")
        emb: Tensor = x[0].to(dtype=torch.float32)
        return self.model(emb)

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
                features: List[Tensor] = [tensor.to(device=dataset.device) for tensor in x]
                targets: Tensor = y.to(dtype=torch.float32, device=dataset.device)
                outputs: Tensor = self.forward(features)
                loss: Tensor = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def save(self, file_path: str) -> None:
        state = {
            "state_dict": self.state_dict(),
            "params": self.params,
            "input_dim": self.input_dim,
        }
        torch.save(state, file_path)

    @staticmethod
    def load(file_path: str) -> EmbeddingOnlyModel:
        checkpoint = torch.load(file_path, map_location="cpu")
        model = EmbeddingOnlyModel(params=checkpoint.get("params", {}))
        input_dim = checkpoint.get("input_dim")
        if input_dim is None:
            raise ValueError("Missing input_dim in checkpoint; cannot reconstruct model.")
        model._build_model(input_dim=input_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
