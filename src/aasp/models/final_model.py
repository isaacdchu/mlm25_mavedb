from __future__ import annotations
from typing import Dict, Any, List, Optional
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

class FinalModel(Model):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params
        self.scoreset_embedding_dim: int = self.params.get('scoreset_embedding_dim', 8)
        self.biotype_embedding_dim: int = self.params.get('biotype_embedding_dim', 8)
        # Attributes to be defined in transform
        self.esm_in_features: int = 0
        self.nn_in_features: int = 0
        self.esm_model: torch.nn.Sequential = torch.nn.Sequential()
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.scoreset_stats: Dict[str, Dict[str, float]] = {}
        self.id_to_scoreset: Dict[int, str] = {}
        self.scoreset_emb: torch.nn.Embedding
        self.biotype_emb: torch.nn.Embedding
        self.initialized: bool = False
        self.generator: Optional[torch.Generator] = None

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Select relevant columns
        features: List[str] = ["relative_pos", "ref_long", "alt_long", "scoreset", "biotype", "ref_embedding", "alt_embedding"]
        if "score" not in data.columns:
            data = data[features]
        else:
            data = data[["score"] + features]
            # Calculate mean and standard deviation for each scoreset
            if not self.scoreset_stats:
                print("Calculating scoreset statistics...")
                for scoreset in data["scoreset"].unique():
                    subset: pd.Series = data[data["scoreset"] == scoreset]["score"]
                    # Remove top and bottom 1% of scores to avoid outliers
                    lower_bound: float = subset.quantile(0.01)
                    upper_bound: float = subset.quantile(0.99)
                    filtered_subset: pd.Series = subset[(subset >= lower_bound) & (subset <= upper_bound)]
                    if len(filtered_subset) <= 20:
                        filtered_subset = subset
                    self.scoreset_stats[scoreset] = {
                        "mean": filtered_subset.mean(),
                        "std": filtered_subset.std() if filtered_subset.std() > 0 else 1.0
                    }
            else:
                print("Using precomputed scoreset statistics...")

        # One-hot encode categorical features
        data = DataHandler.one_hot_encode(data, columns=["ref_long", "alt_long"])

        # Compute embedding distances and similarities
        embedding_distances: List[float] = []
        embedding_similarities: List[float] = []
        embedding_differences: List[np.ndarray] = []
        for _, row in data.iterrows():
            ref: Tensor = torch.nan_to_num(row["ref_embedding"])
            alt: Tensor = torch.nan_to_num(row["alt_embedding"])
            embedding_distances.append(torch.dist(ref, alt, p=2).item())
            embedding_similarities.append(torch.cosine_similarity(ref.unsqueeze(0), alt.unsqueeze(0)).item())
            embedding_differences.append((alt - ref).numpy())

        data["embedding_distance"] = embedding_distances
        data["embedding_similarity"] = embedding_similarities
        data["embedding_difference"] = embedding_differences
        data.drop(columns=["ref_embedding", "alt_embedding"], inplace=True)

        # Embed scoreset
        unique_scoresets: List[str] = sorted(data["scoreset"].unique())
        scoreset_to_id: Dict[str, int] = {name: i for i, name in enumerate(unique_scoresets)}
        self.id_to_scoreset: Dict[int, str] = {i: name for i, name in enumerate(unique_scoresets)}
        data["scoreset_id"] = data["scoreset"].map(scoreset_to_id)
        data.drop(columns=["scoreset"], inplace=True)

        # Embed biotype
        unique_biotypes: List[str] = sorted(data["biotype"].unique())
        biotype_to_id: Dict[str, int] = {name: i for i, name in enumerate(unique_biotypes)}
        data["biotype_id"] = data["biotype"].map(biotype_to_id)
        data.drop(columns=["biotype"], inplace=True)

        # Define the model architecture if uninitialized
        if self.initialized:
            print("model already initialized")
            return data
        self.initialized = True
        self.scoreset_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings=len(unique_scoresets),
            embedding_dim=self.scoreset_embedding_dim
        )
        self.biotype_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings=len(unique_biotypes),
            embedding_dim=self.biotype_embedding_dim
        )

        # Exclude "score" column and replace scoreset with scoreset embedding dimension
        self.esm_in_features:int = len(data["embedding_difference"].iloc[0])
        print(f"ESM input features: {self.esm_in_features}")
        self.nn_in_features:int = data.shape[1] + self.scoreset_embedding_dim + self.biotype_embedding_dim - 4
        self.esm_model = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.esm_in_features, out_channels=128, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64, out_features=32),
        )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.nn_in_features + 32, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=128, out_features=1),
        )
        return data

    def forward(self, x: List[Tensor]) -> Tensor:
        biotype_id_tensor: Tensor = x.pop()
        biotype_embeddings: Tensor = self.biotype_emb.forward(biotype_id_tensor.to(dtype=torch.long))
        scoreset_id_tensor: Tensor = x.pop()
        scoreset_embeddings: Tensor = self.scoreset_emb.forward(scoreset_id_tensor.to(dtype=torch.long))
        embedding_difference_tensor: Tensor = x.pop()
        esm_input_tensor: Tensor = embedding_difference_tensor.squeeze(dim=0).unsqueeze(dim=2)
        esm_features: Tensor = self.esm_model.forward(esm_input_tensor)
        input_tensor: Tensor = torch.stack(x, dim=1)
        input_tensor = torch.cat([input_tensor, scoreset_embeddings, biotype_embeddings, esm_features], dim=1)
        y_pred: Tensor = self.model(input_tensor)
        scoresets: List[str] = [self.id_to_scoreset[int(idx.cpu().detach().numpy().item())] for idx in scoreset_id_tensor]
        scoreset_stats: List[Dict[str, float]] = [self.scoreset_stats[scoreset] for scoreset in scoresets]
        for i, stats in enumerate(scoreset_stats):
            y_pred[i] = y_pred[i] * stats["std"] + stats["mean"]
        return y_pred

    def train_loop(
        self,
        train_dataset: AASPDataset,
        test_dataset: Optional[AASPDataset],
        criterion: Module,
        optimizer: Optimizer,
        params: Dict[str, Any]
    ) -> None:
        if test_dataset is not None and train_dataset.device != test_dataset.device:
            raise ValueError("Train and test datasets must be on the same device")
        device = train_dataset.device
        batch_size: int = abs(params.get('batch_size', 32))
        num_epochs: int = abs(params.get('num_epochs', 10))
        data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=self.generator)
        self.to(device=device)
        self.model.to(device=device)
        self.scoreset_emb.to(device=device)
        for epoch in range(num_epochs):
            # Train batch
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.train()
            for batch_idx, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                x: List[Tensor] = [tensor.to(device=device) for tensor in x]
                y: Tensor = y.to(dtype=torch.float32, device=device)
                outputs: Tensor = self.forward(x)
                loss: Tensor = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # Create inference checkpoint
            save_path: str = params.get("save_path", "output/models/")
            self.save(f"{save_path}/final_model_epoch_{epoch + 1}.pt")

            # Evaluate train and validation loss
            if (test_dataset is None):
                continue
            self.eval()
            with torch.no_grad():
                train_loader: DataLoader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
                (x_train, y_train) = next(iter(train_loader))
                x_train: List[Tensor] = [tensor.to(device=device) for tensor in x_train]
                y_train: Tensor = y_train.to(dtype=torch.float32, device=device)
                train_outputs: Tensor = self.forward(x_train)
                train_loss: Tensor = criterion(train_outputs, y_train)
                print(f"\tTrain Loss: {train_loss.item()}")
            with torch.no_grad():
                test_loader: DataLoader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
                (x_test, y_test) = next(iter(test_loader))
                x_test: List[Tensor] = [tensor.to(device=device) for tensor in x_test]
                y_test: Tensor = y_test.to(dtype=torch.float32, device=device)
                test_outputs: Tensor = self.forward(x_test)
                test_loss: Tensor = criterion(test_outputs, y_test)
                print(f"\tTest Loss: {test_loss.item()}")

    def save(self, file_path: str) -> None:
        if not self.initialized:
            raise ValueError("Model must be initialized before saving")
        torch.save({
            'cnn_state_dict': self.esm_model.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'scoreset_emb_state_dict': self.scoreset_emb.state_dict(),
            'biotype_emb_state_dict': self.biotype_emb.state_dict(),
            'scoreset_embedding_dim': self.scoreset_embedding_dim,
            'biotype_embedding_dim': self.biotype_embedding_dim,
            'scoreset_stats': self.scoreset_stats,
            'id_to_scoreset': self.id_to_scoreset,
            'esm_in_features': self.esm_in_features,
            'nn_in_features': self.nn_in_features,
            'params': self.params
        }, file_path)

    @staticmethod
    def load(file_path: str) -> FinalModel:
        checkpoint = torch.load(file_path, weights_only=False)
        model = FinalModel(params=checkpoint['params'])
        model.initialized = True
        model.scoreset_stats = checkpoint['scoreset_stats']
        model.id_to_scoreset = checkpoint['id_to_scoreset']
        model.esm_in_features = checkpoint['esm_in_features']
        model.nn_in_features = checkpoint['nn_in_features']
        model.esm_model.load_state_dict(checkpoint['cnn_state_dict'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.scoreset_emb.load_state_dict(checkpoint['scoreset_emb_state_dict'])
        model.biotype_emb.load_state_dict(checkpoint['biotype_emb_state_dict'])
        model.scoreset_embedding_dim = checkpoint['scoreset_embedding_dim']
        model.biotype_embedding_dim = checkpoint['biotype_embedding_dim']
        return model

    def set_deterministic(self, seed: int = 0) -> None:
        """
        Set random seeds for reproducibility.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        self.generator = torch.Generator().manual_seed(seed)
