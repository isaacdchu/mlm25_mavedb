"""
ScoresetEnsembleModel
---------------------
Implements the documented Model interface and trains a separate lightweight MLP
for each scoreset. At inference, each sample is routed to its scoreset-specific
model; unseen scoresets fall back to the scoreset mean (or global mean).
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
import math
import pandas as pd
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from .model_interface import Model
from .aasp_dataset import AASPDataset


class _ScoresetMLP(nn.Module):
    """Small MLP used per scoreset."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class ScoresetEnsembleModel(Model):
    """
    Per-scoreset ensemble model. Fits a separate MLP per scoreset ID and keeps
    per-scoreset means as a strong baseline fallback.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.hidden_dim: int = int(params.get("hidden_dim", 128))
        self.dropout: float = float(params.get("dropout", 0.1))
        self.batch_size: int = int(params.get("batch_size", 64))
        self.epochs: int = int(params.get("epochs", 50))
        self.patience: int = int(params.get("patience", 8))
        self.lr: float = float(params.get("lr", 1e-3))
        self.weight_decay: float = float(params.get("weight_decay", 1e-5))
        self.device: str = params.get("device", "cpu")

        self.scoreset_to_id: Dict[str, int] = {}
        self.id_to_scoreset: Dict[int, str] = {}
        self.feature_dim: int | None = None
        self.scoreset_models: nn.ModuleDict = nn.ModuleDict()
        self.scoreset_means: Dict[str, float] = {}
        self.global_mean: float = 0.0
        self.val_sets: Dict[str, Tuple[Tensor, Tensor]] = {}
        self.scoreset_stats: Dict[str, Dict[str, float]] = {}

    def _ensure_encoders(self) -> None:
        if not self.scoreset_to_id:
            raise RuntimeError("Scoreset encoders not initialized. Call transform on training data first.")

    def _fit_or_map(self, values: pd.Series, col: str, fit: bool) -> Tuple[pd.Series, Dict[str, int]]:
        """Fit or apply a simple label mapping for a categorical column."""
        mapping = self.scoreset_to_id if col == "scoreset" else {}
        if fit:
            uniques = sorted(values.fillna("UNKNOWN").unique().tolist())
            mapping.clear()
            mapping.update({v: i for i, v in enumerate(uniques)})
            self.id_to_scoreset = {i: v for v, i in mapping.items()}
        mapped = values.fillna("UNKNOWN").map(mapping).fillna(-1).astype(int)
        return mapped, mapping

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature vectors:
        - diff and abs-diff of embeddings
        - positional scalar
        - encoded categorical scalars for ensp/ref_long/alt_long/biotype
        Adds integer scoreset_id used for routing.
        """
        df = data.copy()
        fit_encoders = len(self.scoreset_to_id) == 0

        # Encode scoreset and other categorical columns
        df["scoreset_id"], _ = self._fit_or_map(df["scoreset"], "scoreset", fit_encoders)

        def fit_map_col(col: str) -> pd.Series:
            key = f"_map_{col}"
            existing = getattr(self, key, None)
            if fit_encoders or existing is None:
                uniques = sorted(df[col].fillna("UNKNOWN").unique().tolist())
                mapping = {v: i for i, v in enumerate(uniques)}
                setattr(self, key, mapping)
            mapping = getattr(self, key)
            return df[col].fillna("UNKNOWN").map(mapping).fillna(-1).astype(int)

        for cat in ["ensp", "ref_long", "alt_long", "biotype"]:
            if cat in df.columns:
                df[f"{cat}_id"] = fit_map_col(cat)
            else:
                df[f"{cat}_id"] = -1

        features: List[np.ndarray] = []
        scoreset_ids: List[int] = []
        targets: List[float] = []

        for _, row in df.iterrows():
            ref_emb = np.nan_to_num(np.array(row["ref_embedding"], dtype=float))
            alt_emb = np.nan_to_num(np.array(row["alt_embedding"], dtype=float))
            if ref_emb.shape != alt_emb.shape:
                continue
            if np.isnan(ref_emb).any() or np.isnan(alt_emb).any():
                continue

            diff = alt_emb - ref_emb
            abs_diff = np.abs(diff)
            extra = np.array(
                [
                    float(row.get("pos", 0.0)),
                    float(row["ensp_id"]),
                    float(row["ref_long_id"]),
                    float(row["alt_long_id"]),
                    float(row["biotype_id"]),
                ],
                dtype=float,
            )
            feat_vec = np.concatenate([diff, abs_diff, extra])

            features.append(feat_vec)
            scoreset_ids.append(int(row["scoreset_id"]))
            targets.append(float(row["score"]))

        processed = pd.DataFrame(
            {
                "score": targets,
                "features": features,
                "scoreset_id": scoreset_ids,
            }
        )

        if self.feature_dim is None and features:
            self.feature_dim = len(features[0])

        return processed

    def prepare_inference(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        Build features and scoreset IDs for inference data (no scores required).
        Returns (features, scoreset_ids, accessions) where accessions will be
        pulled from the input column if present, else row indices.
        """
        self._ensure_encoders()
        df = data.copy()
        df["scoreset_id"], _ = self._fit_or_map(df["scoreset"], "scoreset", fit=False)

        def map_col(col: str) -> pd.Series:
            key = f"_map_{col}"
            mapping = getattr(self, key, None)
            if mapping is None:
                # unseen mapping: assign -1
                return pd.Series([-1] * len(df), index=df.index, dtype=int)
            return df[col].fillna("UNKNOWN").map(mapping).fillna(-1).astype(int)

        for cat in ["ensp", "ref_long", "alt_long", "biotype"]:
            df[f"{cat}_id"] = map_col(cat) if cat in df.columns else -1

        features: List[np.ndarray] = []
        scoreset_ids: List[int] = []
        accessions: List[Any] = []

        for idx, row in df.iterrows():
            ref_emb = np.nan_to_num(np.array(row["ref_embedding"], dtype=float))
            alt_emb = np.nan_to_num(np.array(row["alt_embedding"], dtype=float))
            if ref_emb.shape != alt_emb.shape:
                continue
            if np.isnan(ref_emb).any() or np.isnan(alt_emb).any():
                continue

            diff = alt_emb - ref_emb
            abs_diff = np.abs(diff)
            extra = np.array(
                [
                    float(row.get("pos", 0.0)),
                    float(row["ensp_id"]),
                    float(row["ref_long_id"]),
                    float(row["alt_long_id"]),
                    float(row["biotype_id"]),
                ],
                dtype=float,
            )
            feat_vec = np.concatenate([diff, abs_diff, extra])
            features.append(feat_vec)
            scoreset_ids.append(int(row["scoreset_id"]))
            accessions.append(row["accession"] if "accession" in row else idx)

        return np.array(features), np.array(scoreset_ids), accessions

    def _build_grouped_tensors(
        self, dataset: AASPDataset
    ) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Group dataset samples by scoreset_id for per-scoreset training."""
        grouped: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
        for i in range(len(dataset)):
            x_list, y = dataset[i]
            feats: Tensor = x_list[0].to(torch.float32)
            scoreset_id: int = int(x_list[1].item())
            grouped.setdefault(scoreset_id, []).append((feats, y.squeeze(-1)))

        grouped_tensors: Dict[int, Tuple[Tensor, Tensor]] = {}
        for sid, rows in grouped.items():
            feats = torch.stack([r[0] for r in rows])
            labels = torch.stack([r[1] for r in rows])
            grouped_tensors[sid] = (feats, labels)
        return grouped_tensors

    def forward(self, x: List[Tensor]) -> Tensor:
        """
        Route each sample to its scoreset-specific model. If a scoreset model is
        missing, fall back to the scoreset mean, then to the global mean.
        """
        features, scoreset_ids = x[0], x[1]
        if features.dim() == 1:
            features = features.unsqueeze(0)
        outputs: List[Tensor] = []
        device = features.device
        for feats, sid_tensor in zip(features, scoreset_ids):
            sid = int(sid_tensor.item())
            sid_key = str(sid)
            if sid_key in self.scoreset_models:
                pred = self.scoreset_models[sid_key](feats.unsqueeze(0)).squeeze(0)
                # Clip to observed label range for that scoreset to avoid blowups
                stats = self.scoreset_stats.get(sid_key, {})
                if stats:
                    lbl_min = stats.get("min", -float("inf"))
                    lbl_max = stats.get("max", float("inf"))
                    pred = torch.clamp(pred, min=lbl_min, max=lbl_max)
            else:
                mean_val = self.scoreset_means.get(sid_key, self.global_mean)
                pred = torch.tensor(mean_val, device=device, dtype=feats.dtype)
            outputs.append(pred)
        return torch.stack(outputs)

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: nn.Module,
        optimizer: Optional[Optimizer],
        params: Dict[str, Any],
    ) -> None:
        """
        Train a separate MLP per scoreset using early stopping on a val split.
        Optimizer from params is unused because we need per-model optimizers.
        """
        batch_size = int(params.get("batch_size", self.batch_size))
        epochs = int(params.get("epochs", self.epochs))
        patience = int(params.get("patience", self.patience))
        lr = float(params.get("lr", self.lr))
        weight_decay = float(params.get("weight_decay", self.weight_decay))
        device = self.device
        max_mse_threshold = float(params.get("max_mse_threshold", float("inf")))

        grouped = self._build_grouped_tensors(dataset)
        all_labels = torch.cat([labels for _, labels in grouped.values()])
        self.global_mean = float(all_labels.mean().item()) if len(all_labels) else 0.0

        self.scoreset_models = nn.ModuleDict()
        self.scoreset_means = {}
        self.val_sets = {}
        self.scoreset_stats = {}

        for sid, (feats, labels) in tqdm(grouped.items(), desc="Training scoresets"):
            sid_key = str(sid)
            self.scoreset_means[sid_key] = float(labels.mean().item())

            if feats.numel() == 0:
                continue

            model = _ScoresetMLP(self.feature_dim or feats.shape[1], self.hidden_dim, self.dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Simple train/val split
            n = feats.shape[0]
            perm = torch.randperm(n)
            split = max(1, int(0.2 * n))
            val_idx = perm[:split]
            train_idx = perm[split:] if split < n else perm
            train_data = TensorDataset(feats[train_idx].to(device), labels[train_idx].to(device))
            val_data = TensorDataset(feats[val_idx].to(device), labels[val_idx].to(device))
            # Store val data for later reporting
            self.val_sets[sid_key] = (feats[val_idx].to(device), labels[val_idx].to(device))
            # Track label stats for clipping/fallback
            lbl_min = float(labels.min().item())
            lbl_max = float(labels.max().item())
            lbl_mean = float(labels.mean().item())

            train_loader = DataLoader(train_data, batch_size=min(batch_size, len(train_data)), shuffle=True, drop_last=False)
            val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False) if len(val_data) else None

            best_loss = math.inf
            patience_ctr = 0
            best_state: Dict[str, Tensor] | None = None

            for _ in range(epochs):
                model.train()
                for xb, yb in train_loader:
                    opt.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                if val_loader is None:
                    continue
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for xb, yb in val_loader:
                        preds = model(xb)
                        val_losses.append(criterion(preds, yb).item())
                    val_loss = float(np.mean(val_losses)) if val_losses else math.inf
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            # Final val metrics on stored val set
            if val_data:
                with torch.no_grad():
                    xb, yb = self.val_sets[sid_key]
                    preds_val = model(xb)
                    val_mse = float(torch.mean((preds_val - yb) ** 2).item())
                    val_mae = float(torch.mean(torch.abs(preds_val - yb)).item())
            else:
                val_mse, val_mae = float("inf"), float("inf")

            self.scoreset_stats[sid_key] = {
                "n": float(n),
                "mean": lbl_mean,
                "min": lbl_min,
                "max": lbl_max,
                "val_mse": val_mse,
                "val_mae": val_mae,
            }

            # If validation is terrible, skip this model and fall back to mean
            if val_mse > max_mse_threshold:
                continue

            self.scoreset_models[sid_key] = model

    def save(self, file_path: str) -> None:
        torch.save(
            {
                "state_dicts": {k: v.state_dict() for k, v in self.scoreset_models.items()},
                "scoreset_means": self.scoreset_means,
                "global_mean": self.global_mean,
                "feature_dim": self.feature_dim,
                "scoreset_to_id": self.scoreset_to_id,
                "id_to_scoreset": self.id_to_scoreset,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "scoreset_stats": self.scoreset_stats,
            },
            file_path,
        )

    @staticmethod
    def load(file_path: str) -> "ScoresetEnsembleModel":
        checkpoint = torch.load(file_path, map_location="cpu")
        model = ScoresetEnsembleModel(
            {
                "hidden_dim": checkpoint.get("hidden_dim", 128),
                "dropout": checkpoint.get("dropout", 0.1),
            }
        )
        model.feature_dim = checkpoint.get("feature_dim")
        model.scoreset_means = checkpoint.get("scoreset_means", {})
        model.global_mean = checkpoint.get("global_mean", 0.0)
        model.scoreset_to_id = checkpoint.get("scoreset_to_id", {})
        model.id_to_scoreset = checkpoint.get("id_to_scoreset", {})
        model.scoreset_stats = checkpoint.get("scoreset_stats", {})
        state_dicts = checkpoint.get("state_dicts", {})
        for sid_key, sd in state_dicts.items():
            mlp = _ScoresetMLP(model.feature_dim or 1, model.hidden_dim, model.dropout)
            mlp.load_state_dict(sd)
            model.scoreset_models[sid_key] = mlp
        return model
