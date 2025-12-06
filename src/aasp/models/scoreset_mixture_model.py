from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from .model_interface import Model
from .aasp_dataset import AASPDataset
from .data_handler import DataHandler


class ScoresetMixtureModel(Model):
    """
    Scoreset-specialized mixture: each scoreset gets its own small MLP.
    Routes samples to the corresponding submodel based on scoreset_id.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: Dict[str, Any] = params
        self.submodels: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.scoreset_to_id: Dict[str, int] = {}
        self.id_to_scoreset: Dict[int, str] = {}
        self.submodel_input_dim: int | None = None

    def _ensure_rel_pos(self, data: pd.DataFrame) -> pd.DataFrame:
        if "rel_pos" in data.columns:
            return data
        if "pos" in data.columns:
            max_pos: float = float(np.nanmax(data["pos"].astype(float)))
            if max_pos > 0:
                data = data.copy()
                data["rel_pos"] = data["pos"].astype(float) / max_pos
        return data

    def _build_submodel(self, input_dim: int) -> torch.nn.Sequential:
        hidden: int = int(self.params.get("hidden_dim", 128))
        hidden2: int = int(self.params.get("hidden_dim_2", 64))
        dropout: float = float(self.params.get("dropout", 0.1))
        activation = torch.nn.GELU() if self.params.get("use_gelu", True) else torch.nn.ReLU()
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden),
            activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=hidden, out_features=hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden2, out_features=1)
        )

    def _get_or_create_submodel(self, scoreset_id: int, input_dim: int) -> torch.nn.Sequential:
        key = str(scoreset_id)
        if key not in self.submodels:
            self.submodels[key] = self._build_submodel(input_dim)
        return self.submodels[key]

    def _stack_features(self, features: List[Tensor]) -> Tensor:
        if not features:
            raise ValueError("Feature list is empty.")
        if len(features[0].shape) > 1:
            return torch.cat(features, dim=-1).to(dtype=torch.float32)
        return torch.stack(features, dim=0).mT.to(dtype=torch.float32)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if "scoreset" not in data.columns:
            raise KeyError("'scoreset' column is required for ScoresetMixtureModel.")
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
            ]
            if col in data.columns
        ]
        if cat_cols:
            data = DataHandler.one_hot_encode(data, columns=cat_cols)

        unique_scoresets: List[str] = sorted(data["scoreset"].unique())
        self.scoreset_to_id = {name: idx for idx, name in enumerate(unique_scoresets)}
        self.id_to_scoreset = {idx: name for name, idx in self.scoreset_to_id.items()}
        data["scoreset_id"] = data["scoreset"].map(self.scoreset_to_id)

        drop_cols = [col for col in ["accession", "ensp", "scoreset", "ref_embedding", "alt_embedding", "emb_diff"] if col in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols)

        feature_cols: List[str] = [col for col in data.columns if col != "score"]
        feature_cols = [col for col in feature_cols if col != "scoreset_id"] + ["scoreset_id"]
        data = data[["score"] + feature_cols]

        total_features: int = data.shape[1] - 1
        self.submodel_input_dim = total_features - 1  # reserve last feature for scoreset_id
        if self.submodel_input_dim <= 0:
            raise ValueError("Insufficient features to train submodels.")
        for scoreset_id in self.scoreset_to_id.values():
            self._get_or_create_submodel(scoreset_id, input_dim=self.submodel_input_dim)
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        if not x:
            raise ValueError("Input feature list is empty.")
        scoreset_ids: Tensor = x[-1].to(dtype=torch.long)
        features: List[Tensor] = [tensor for tensor in x[:-1]]
        feature_tensor: Tensor = self._stack_features(features)
        device = scoreset_ids.device
        preds: Tensor = torch.empty((scoreset_ids.shape[0], 1), device=device, dtype=torch.float32)
        for sid in torch.unique(scoreset_ids):
            sid_int = int(sid.item())
            mask: Tensor = scoreset_ids == sid
            submodel = self._get_or_create_submodel(
                scoreset_id=sid_int,
                input_dim=self.submodel_input_dim or feature_tensor.shape[1]
            ).to(device)
            preds[mask] = submodel(feature_tensor[mask])
        return preds

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        batch_size: int = abs(params.get("batch_size", 32))
        num_epochs: int = abs(params.get("num_epochs", 10))
        self.train()
        self.to(device=dataset.device)

        # group indices by scoreset id
        groupings: Dict[int, List[int]] = {}
        for idx, sample in enumerate(dataset.x):
            sid = int(sample[-1].item())
            groupings.setdefault(sid, []).append(idx)
            if str(sid) not in self.submodels:
                self._get_or_create_submodel(
                    sid,
                    input_dim=self.submodel_input_dim or (len(sample) - 1)
                )

        for _ in range(num_epochs):
            for sid, indices in groupings.items():
                sub_dataset: Subset[AASPDataset] = Subset(dataset, indices)
                loader: DataLoader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=True)
                submodel = self.submodels[str(sid)].to(device=dataset.device)
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    features = [tensor.to(device=dataset.device, dtype=torch.float32) for tensor in x_batch[:-1]]
                    preds: Tensor = submodel(self._stack_features(features))
                    targets: Tensor = y_batch.to(dtype=torch.float32, device=dataset.device)
                    loss: Tensor = criterion(preds, targets)
                    loss.backward()
                    optimizer.step()

    def save(self, file_path: str) -> None:
        state = {
            "state_dict": self.state_dict(),
            "params": self.params,
            "scoreset_to_id": self.scoreset_to_id,
            "submodel_input_dim": self.submodel_input_dim,
        }
        torch.save(state, file_path)

    @staticmethod
    def load(file_path: str) -> ScoresetMixtureModel:
        checkpoint = torch.load(file_path, map_location="cpu")
        model = ScoresetMixtureModel(params=checkpoint.get("params", {}))
        model.scoreset_to_id = checkpoint.get("scoreset_to_id", {})
        model.id_to_scoreset = {idx: name for name, idx in model.scoreset_to_id.items()}
        model.submodel_input_dim = checkpoint.get("submodel_input_dim")
        if model.submodel_input_dim is None:
            raise ValueError("Missing submodel_input_dim in checkpoint; cannot reconstruct model.")
        for sid in model.scoreset_to_id.values():
            model._get_or_create_submodel(sid, input_dim=model.submodel_input_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
