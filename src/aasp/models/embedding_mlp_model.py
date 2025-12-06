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


class EmbeddingMLPModel(Model):
    """
    MLP over embedding difference + tabular features (Feature Set B).
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: Dict[str, Any] = params
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.input_dim: int | None = None

    def _ensure_rel_pos(self, data: pd.DataFrame) -> pd.DataFrame:
        if "rel_pos" in data.columns:
            return data
        if "pos" in data.columns:
            max_pos: float = float(np.nanmax(data["pos"].astype(float)))
            if max_pos > 0:
                data = data.copy()
                data["rel_pos"] = data["pos"].astype(float) / max_pos
        return data

    def _build_model(self, input_dim: int) -> None:
        widths: List[int] = self.params.get("hidden_layers", [256, 128, 64])
        activations: List[Module] = []
        for _ in widths:
            activations.append(torch.nn.GELU() if self.params.get("use_gelu", True) else torch.nn.ReLU())
        layers: List[Module] = []
        in_dim = input_dim
        for width, act in zip(widths, activations):
            layers.append(torch.nn.Linear(in_features=in_dim, out_features=width))
            layers.append(act)
            layers.append(torch.nn.Dropout(float(self.params.get("dropout", 0.1))))
            in_dim = width
        layers.append(torch.nn.Linear(in_features=in_dim, out_features=1))
        self.model = torch.nn.Sequential(*layers)
        self.input_dim = input_dim

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if "ref_embedding" not in data.columns or "alt_embedding" not in data.columns:
            raise KeyError("Both 'ref_embedding' and 'alt_embedding' columns are required for EmbeddingMLPModel.")
        data = data.copy()
        data = self._ensure_rel_pos(data)
        def _compute_diff(row: pd.Series) -> Tensor:
            ref: Tensor = torch.as_tensor(row["ref_embedding"], dtype=torch.float32)
            alt: Tensor = torch.as_tensor(row["alt_embedding"], dtype=torch.float32)
            return torch.nan_to_num(alt - ref)
        data["emb_diff"] = data.apply(_compute_diff, axis=1)
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
        drop_cols = [col for col in ["accession", "ensp", "ref_embedding", "alt_embedding"] if col in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols)

        embed_dim: int = len(data["emb_diff"].iloc[0])
        scalar_feature_count: int = data.shape[1] - 2  # exclude score and emb_diff
        input_dim: int = embed_dim + scalar_feature_count
        self._build_model(input_dim=input_dim)

        feature_cols: List[str] = [col for col in data.columns if col != "score"]
        feature_cols = ["emb_diff"] + [col for col in feature_cols if col != "emb_diff"]
        data = data[["score"] + feature_cols]
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        if not x:
            raise ValueError("Input feature list is empty.")
        emb: Tensor = x[0].to(dtype=torch.float32)
        scalar_tensors: List[Tensor] = [
            tensor.unsqueeze(-1) if tensor.dim() == 1 else tensor
            for tensor in x[1:]
        ]
        if scalar_tensors:
            scalar_features: Tensor = torch.cat(
                [t.to(dtype=torch.float32) for t in scalar_tensors],
                dim=1
            )
            input_tensor: Tensor = torch.cat([emb, scalar_features], dim=1)
        else:
            input_tensor = emb
        return self.model(input_tensor)

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
    def load(file_path: str) -> EmbeddingMLPModel:
        checkpoint = torch.load(file_path, map_location="cpu")
        model = EmbeddingMLPModel(params=checkpoint.get("params", {}))
        input_dim = checkpoint.get("input_dim")
        if input_dim is None:
            raise ValueError("Missing input_dim in checkpoint; cannot reconstruct model.")
        model._build_model(input_dim=input_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
