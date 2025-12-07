from __future__ import annotations
from typing import Dict, Any, List, Dict
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

class ExperimentModel(Model):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.embedding_dim: int = params.get('scoreset_embedding_dim', 8)
        self.model: torch.nn.Sequential = torch.nn.Sequential() # to be defined in transform
        self.scoreset_emb: torch.nn.Embedding # to be defined in transform

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Select relevant columns
        data = data[["score", "pos", "ref_long", "alt_long", "scoreset", "ref_embedding", "alt_embedding"]]

        # Convert "pos" to float
        data.loc[:, "pos"] = data["pos"].astype(float)

        # One-hot encode categorical features
        data = DataHandler.one_hot_encode(data, columns=["ref_long", "alt_long"])

        # Compute embedding distances and similarities
        def compute_distances(row: pd.Series) -> pd.Series:
            ref: Tensor = torch.nan_to_num(row["ref_embedding"])
            alt: Tensor = torch.nan_to_num(row["alt_embedding"])
            row["embedding_distance"] = torch.dist(ref, alt, p=2).item()
            return row
        def compute_similarity(row: pd.Series) -> pd.Series:
            ref: Tensor = torch.nan_to_num(row["ref_embedding"])
            alt: Tensor = torch.nan_to_num(row["alt_embedding"])
            row["embedding_similarity"] = torch.cosine_similarity(
                ref.unsqueeze(0),
                alt.unsqueeze(0)
            ).item()
            return row
        data = data.apply(compute_distances, axis=1)
        data = data.apply(compute_similarity, axis=1)
        data.drop(columns=["ref_embedding", "alt_embedding"], inplace=True)

        # Embed scoreset
        unique_scoresets: List[str] = sorted(data["scoreset"].unique())
        scoreset_to_id: Dict[str, int] = {name: i for i, name in enumerate(unique_scoresets)}
        data["scoreset_id"] = data["scoreset"].map(scoreset_to_id)
        data.drop(columns=["scoreset"], inplace=True)
        
        # Define the model architecture
        self.scoreset_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings=len(unique_scoresets),
            embedding_dim=self.embedding_dim
        )

        # Exclude "score" column and replace scoreset with scoreset embedding dimension
        in_features: int = data.shape[1] + self.embedding_dim - 2 
        class InverseTanh(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return 0.5 * (torch.exp(2 * x) - 1) / (torch.exp(2 * x) + 1)
        self.model: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=in_features, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
            InverseTanh()
        )
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        scoreset_id_tensor: Tensor = x[-1]
        scoreset_embeddings: Tensor = self.scoreset_emb.forward(scoreset_id_tensor.to(dtype=torch.long))
        x.pop(-1)
        input_tensor: Tensor = torch.stack(x, dim=1)
        input_tensor = torch.cat([input_tensor, scoreset_embeddings], dim=1)
        return self.model(input_tensor)

    def train_loop(
        self,
        train_dataset: AASPDataset,
        test_dataset: AASPDataset,
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        batch_size: int = abs(params.get('batch_size', 32))
        num_epochs: int = abs(params.get('num_epochs', 10))
        if train_dataset.device != test_dataset.device:
            raise ValueError("Train and test datasets must be on the same device")
        device = train_dataset.device
        data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.train()
        self.to(device=device)
        self.model.to(device=device)
        self.scoreset_emb.to(device=device)
        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                x: List[Tensor] = [tensor.to(device=device) for tensor in x]
                y: Tensor = y.to(dtype=torch.float32, device=device)
                outputs: Tensor = self.forward(x)
                loss: Tensor = criterion(outputs, y)
                loss.backward()
                optimizer.step()

    def save(self, file_path: str) -> None:
        return

    @staticmethod
    def load(file_path: str) -> ExperimentModel:
        return ExperimentModel(params={})
