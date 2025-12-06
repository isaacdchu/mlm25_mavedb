"""
ScoresetShrinkageEnsembleModel
-------------------------------
Implements the Model interface with a global MLP plus per-scoreset MLPs.
At inference, predictions are a shrinkage blend between the global model
and the scoreset-specific model using:
    w_s = n_s / (n_s + shrinkage_alpha).
This stabilizes tiny, high-variance scoresets while still allowing larger
scoresets to learn their own behavior.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
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
    """Small MLP used for both global and per-scoreset models."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class ScoresetShrinkageEnsembleModel(Model):
    """
    Global MLP blended with per-scoreset MLPs using shrinkage weights:
        final = w * local + (1 - w) * global,  w = n / (n + alpha)
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        input_dim = params.get("input_dim")  # can be None; set on first transform
        hidden_dim = int(params.get("hidden_dim", 64))
        dropout = float(params.get("dropout", 0.1))
        device = params.get("device")
        shrinkage_alpha = float(params.get("shrinkage_alpha", 50.0))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_dim: Optional[int] = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.shrinkage_alpha = shrinkage_alpha

        self.global_model: Optional[_ScoresetMLP] = None
        if self.input_dim:
            self.global_model = _ScoresetMLP(self.input_dim, hidden_dim, dropout).to(self.device)

        self.scoreset_models: Dict[Any, _ScoresetMLP] = {}
        self.scoreset_counts: Dict[Any, int] = {}
        self.scoreset_means: Dict[Any, float] = {}
        self.global_mean: float = 0.0
        self.scoreset_to_id: Dict[Any, int] = {}
        self.id_to_scoreset: Dict[int, Any] = {}
        self.val_sets: Dict[Any, Tuple[Tensor, Tensor]] = {}

    # ---------- Utility ----------
    def _ensure_global_model(self, input_dim: int) -> None:
        if self.global_model is None:
            self.input_dim = input_dim
            self.global_model = _ScoresetMLP(input_dim, self.hidden_dim, self.dropout).to(self.device)

    def _build_features(self, df: pd.DataFrame) -> Tuple[Tensor, Tensor, List[int]]:
        """
        Create feature vectors: diff and abs diff of embeddings.
        Encodes scoreset to integer IDs.
        Returns tensors on CPU; caller can move to device.
        """
        features: List[np.ndarray] = []
        targets: List[float] = []
        scoreset_ids: List[int] = []

        for _, row in df.iterrows():
            ref_emb = np.nan_to_num(np.array(row["ref_embedding"], dtype=float))
            alt_emb = np.nan_to_num(np.array(row["alt_embedding"], dtype=float))
            if ref_emb.shape != alt_emb.shape:
                continue
            diff = alt_emb - ref_emb
            abs_diff = np.abs(diff)
            feat_vec = np.concatenate([diff, abs_diff])
            features.append(feat_vec)
            targets.append(float(row["score"]))

            sid_raw = row["scoreset"]
            if sid_raw not in self.scoreset_to_id:
                new_id = len(self.scoreset_to_id)
                self.scoreset_to_id[sid_raw] = new_id
                self.id_to_scoreset[new_id] = sid_raw
            scoreset_ids.append(self.scoreset_to_id[sid_raw])

        X = torch.tensor(np.stack(features), dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        if self.input_dim is None:
            self._ensure_global_model(X.shape[1])
        return X, y, scoreset_ids

    def _train_mlp(self, model: nn.Module, loader: DataLoader, epochs: int, criterion: nn.Module) -> None:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

    # ---------- Interface-required methods ----------
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build a reduced DataFrame containing score, features, and scoreset_id.
        """
        df = data.copy()
        features: List[np.ndarray] = []
        scoreset_ids: List[int] = []
        targets: List[float] = []

        for _, row in df.iterrows():
            ref_emb = np.nan_to_num(np.array(row["ref_embedding"], dtype=float))
            alt_emb = np.nan_to_num(np.array(row["alt_embedding"], dtype=float))
            if ref_emb.shape != alt_emb.shape:
                continue
            diff = alt_emb - ref_emb
            abs_diff = np.abs(diff)
            feat_vec = np.concatenate([diff, abs_diff])
            features.append(feat_vec)
            targets.append(float(row["score"]))

            sid_raw = row["scoreset"]
            if sid_raw not in self.scoreset_to_id:
                new_id = len(self.scoreset_to_id)
                self.scoreset_to_id[sid_raw] = new_id
                self.id_to_scoreset[new_id] = sid_raw
            scoreset_ids.append(self.scoreset_to_id[sid_raw])

        if not features:
            return pd.DataFrame(columns=["score", "features", "scoreset_id"])

        if self.input_dim is None:
            self._ensure_global_model(len(features[0]))

        out_df = pd.DataFrame(
            {
                "score": targets,
                "features": features,
                "scoreset_id": scoreset_ids,
            }
        )
        return out_df

    def forward(self, x: List[Tensor]) -> Tensor:
        """
        x[0]: features tensor, x[1]: scoreset_ids (or raw scoreset values).
        Uses shrinkage blend between global and per-scoreset models.
        """
        feats, scoreset_ids = x[0].to(self.device), x[1]
        if scoreset_ids.dim() == 0:
            scoreset_ids = scoreset_ids.unsqueeze(0)
        if self.global_model is None:
            self._ensure_global_model(feats.shape[1])
        # global predictions (eval mode to avoid BN issues with batch=1)
        self.global_model.eval()
        with torch.no_grad():
            g_preds = self.global_model(feats)
        outputs: List[Tensor] = []
        for i, sid_tensor in enumerate(scoreset_ids):
            sid = sid_tensor.item()
            g = g_preds[i]
            if sid in self.scoreset_models:
                local_model = self.scoreset_models[sid]
                local_model.eval()
                with torch.no_grad():
                    l = local_model(feats[i : i + 1]).squeeze(0)
                n_s = self.scoreset_counts.get(sid, 0)
                w = n_s / (n_s + self.shrinkage_alpha) if n_s > 0 else 0.0
                y_hat = w * l + (1.0 - w) * g
            else:
                # fallback to global prediction (or global mean if needed)
                y_hat = g if torch.isfinite(g) else torch.tensor(self.global_mean, device=self.device)
            outputs.append(y_hat)
        return torch.stack(outputs)

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: nn.Module,
        optimizer: Optional[Optimizer],
        params: Dict[str, Any],
    ) -> None:
        """
        Trains the global model on all samples, then trains per-scoreset models.
        """
        batch_size = int(params.get("batch_size", 128))
        epochs = int(params.get("epochs", 20))
        min_local = int(params.get("min_local_samples", 5))
        self.val_sets = {}

        # Extract tensors from dataset
        feats_list: List[Tensor] = []
        scoreset_list: List[Any] = []
        targets_list: List[Tensor] = []
        for i in range(len(dataset)):
            x_row, y = dataset[i]
            feats_list.append(x_row[0].to(torch.float32))
            scoreset_list.append(int(x_row[1].item()))
            targets_list.append(y.squeeze(-1).to(torch.float32))

        X_all = torch.stack(feats_list).to(self.device)
        y_all = torch.stack(targets_list).to(self.device)
        if self.global_model is None:
            self._ensure_global_model(X_all.shape[1])

        self.global_mean = float(y_all.mean().item()) if len(y_all) else 0.0
        # compute counts and means
        counts: Dict[Any, int] = {}
        sums: Dict[Any, float] = {}
        for sid, y in zip(scoreset_list, y_all):
            counts[sid] = counts.get(sid, 0) + 1
            sums[sid] = sums.get(sid, 0.0) + float(y.item())
        self.scoreset_counts = counts
        self.scoreset_means = {sid: sums[sid] / counts[sid] for sid in counts}

        # Train global model
        global_ds = TensorDataset(X_all, y_all)
        global_loader = DataLoader(global_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        self._train_mlp(self.global_model, global_loader, epochs, criterion)

        # Train per-scoreset models
        self.scoreset_models = {}
        for sid in counts:
            n_s = counts[sid]
            if n_s < min_local:
                continue
            mask = torch.tensor([s == sid for s in scoreset_list], device=self.device)
            X_s = X_all[mask]
            y_s = y_all[mask]
            # simple val split 80/20 within this scoreset
            n = X_s.shape[0]
            perm = torch.randperm(n, device=self.device)
            split = max(1, int(0.2 * n))
            val_idx = perm[:split]
            train_idx = perm[split:] if split < n else perm
            X_train, y_train = X_s[train_idx], y_s[train_idx]
            X_val, y_val = X_s[val_idx], y_s[val_idx]
            self.val_sets[sid] = (X_val, y_val)

            local_model = _ScoresetMLP(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
            ds = TensorDataset(X_train, y_train)
            loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)
            self._train_mlp(local_model, loader, epochs, criterion)
            self.scoreset_models[sid] = local_model

    def save(self, file_path: str) -> None:
        torch.save(
            {
                "global_state": self.global_model.state_dict(),
                "scoreset_states": {sid: m.state_dict() for sid, m in self.scoreset_models.items()},
                "scoreset_counts": self.scoreset_counts,
                "scoreset_means": self.scoreset_means,
                "global_mean": self.global_mean,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "shrinkage_alpha": self.shrinkage_alpha,
                "scoreset_to_id": self.scoreset_to_id,
                "id_to_scoreset": self.id_to_scoreset,
            },
            file_path,
        )

    @staticmethod
    def load(file_path: str) -> "ScoresetShrinkageEnsembleModel":
        checkpoint = torch.load(file_path, map_location="cpu")
        model = ScoresetShrinkageEnsembleModel(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 64),
            dropout=checkpoint.get("dropout", 0.1),
            device="cpu",
            shrinkage_alpha=checkpoint.get("shrinkage_alpha", 50.0),
        )
        model.global_model.load_state_dict(checkpoint["global_state"])
        model.scoreset_counts = checkpoint.get("scoreset_counts", {})
        model.scoreset_means = checkpoint.get("scoreset_means", {})
        model.global_mean = checkpoint.get("global_mean", 0.0)
        model.scoreset_to_id = checkpoint.get("scoreset_to_id", {})
        model.id_to_scoreset = checkpoint.get("id_to_scoreset", {})
        scoreset_states = checkpoint.get("scoreset_states", {})
        for sid, state in scoreset_states.items():
            local = _ScoresetMLP(model.input_dim, model.hidden_dim, model.dropout)
            local.load_state_dict(state)
            model.scoreset_models[sid] = local
        return model

    # ---------- Convenience wrappers ----------
    def fit(self, dataset: AASPDataset, **kwargs: Any) -> None:
        """Alias to train_loop for parity with other code paths."""
        self.train_loop(dataset, nn.MSELoss(), None, kwargs)

    def predict(self, dataset: AASPDataset) -> np.ndarray:
        """
        Predict on an AASPDataset using shrinkage blend.
        """
        feats_list: List[Tensor] = []
        scoreset_list: List[Any] = []
        for i in range(len(dataset)):
            x_row, _ = dataset[i]
            feats_list.append(x_row[0].to(torch.float32))
            scoreset_list.append(int(x_row[1].item()))
        X_all = torch.stack(feats_list).to(self.device)
        if self.global_model is None:
            self._ensure_global_model(X_all.shape[1])
        with torch.no_grad():
            global_pred = self.global_model(X_all).cpu().numpy()

        preds: List[float] = []
        for i, sid in enumerate(scoreset_list):
            g = global_pred[i]
            if sid in self.scoreset_models:
                local_model = self.scoreset_models[sid]
                with torch.no_grad():
                    l = local_model(X_all[i : i + 1]).item()
                n_s = self.scoreset_counts.get(sid, 0)
                w = n_s / (n_s + self.shrinkage_alpha) if n_s > 0 else 0.0
                y_hat = w * l + (1.0 - w) * g
            else:
                y_hat = g if math.isfinite(g) else self.global_mean
            preds.append(float(y_hat))
        return np.array(preds, dtype=float)

    def prepare_inference(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        Build features and scoreset IDs for inference (no scores required).
        Returns (features, scoreset_ids, accessions).
        """
        features: List[np.ndarray] = []
        scoreset_ids: List[int] = []
        accessions: List[Any] = []

        for idx, row in data.iterrows():
            ref_emb = np.nan_to_num(np.array(row["ref_embedding"], dtype=float))
            alt_emb = np.nan_to_num(np.array(row["alt_embedding"], dtype=float))
            if ref_emb.shape != alt_emb.shape:
                continue
            diff = alt_emb - ref_emb
            abs_diff = np.abs(diff)
            feat_vec = np.concatenate([diff, abs_diff])
            features.append(feat_vec)

            sid_raw = row["scoreset"]
            sid = self.scoreset_to_id.get(sid_raw)
            if sid is None:
                sid = len(self.scoreset_to_id)
                self.scoreset_to_id[sid_raw] = sid
                self.id_to_scoreset[sid] = sid_raw
            scoreset_ids.append(sid)

            accessions.append(row["accession"] if "accession" in row else idx)

        return np.array(features), np.array(scoreset_ids), accessions
