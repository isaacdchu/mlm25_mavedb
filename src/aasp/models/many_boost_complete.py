"""
many_boost_complete.py
----------------------
Same as many_boost.py, but intended for final training on 100% of the data
and emitting predictions directly to output.csv without holding out validation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

try:
    from xgboost import XGBRegressor
except ImportError as e:
    XGBRegressor = None  # type: ignore

from .model_interface import Model
from .aasp_dataset import AASPDataset
from tqdm import tqdm


class ManyBoostCompleteModel(Model):
    """
    Trains one XGBoost regressor per scoreset on the full dataset (no val split).
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.device: str = params.get("device", "cpu")
        self.n_estimators: int = int(params.get("n_estimators", 200))
        self.max_depth: int = int(params.get("max_depth", 6))
        self.learning_rate: float = float(params.get("learning_rate", 0.05))
        self.subsample: float = float(params.get("subsample", 0.8))
        self.colsample: float = float(params.get("colsample_bytree", 0.8))
        self.min_child_weight: float = float(params.get("min_child_weight", 1.0))
        self.reg_alpha: float = float(params.get("reg_alpha", 0.0))
        self.reg_lambda: float = float(params.get("reg_lambda", 1.0))
        self.min_local: int = int(params.get("min_local_samples", 5))
        self.n_jobs: int = int(params.get("n_jobs", 4))

        self.scoreset_models: Dict[Any, Any] = {}
        self.scoreset_means: Dict[Any, float] = {}
        self.global_mean: float = 0.0

        self.encoders: Dict[str, Dict[Any, int]] = {}
        self.feature_names: List[str] = []

    def _encode_cat(self, series: pd.Series, col: str, fit: bool) -> pd.Series:
        mapping = self.encoders.get(col, {})
        if fit or not mapping:
            mapping = {val: idx for idx, val in enumerate(sorted(series.fillna("UNKNOWN").unique().tolist()))}
            self.encoders[col] = mapping
        return series.fillna("UNKNOWN").map(mapping).fillna(-1).astype(int)

    def _build_feature_row(self, row: pd.Series) -> Optional[np.ndarray]:
        ref_emb = np.array(row["ref_embedding"], dtype=float)
        alt_emb = np.array(row["alt_embedding"], dtype=float)
        if ref_emb.shape != alt_emb.shape:
            return None
        if np.isnan(ref_emb).any() or np.isnan(alt_emb).any() or np.isinf(ref_emb).any() or np.isinf(alt_emb).any():
            return None
        diff = alt_emb - ref_emb
        abs_diff = np.abs(diff)
        extra = np.array(
            [
                float(row.get("pos", 0.0)),
                float(row.get("scoreset_id", -1)),
                float(row.get("ensp_id", -1)),
                float(row.get("ref_long_id", -1)),
                float(row.get("alt_long_id", -1)),
                float(row.get("biotype_id", -1)),
            ],
            dtype=float,
        )
        return np.concatenate([diff, abs_diff, extra])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        fit_enc = not self.encoders
        df["scoreset_id"] = self._encode_cat(df["scoreset"], "scoreset", fit=fit_enc)
        for cat in ["ensp", "ref_long", "alt_long", "biotype"]:
            if cat in df.columns:
                df[f"{cat}_id"] = self._encode_cat(df[cat], cat, fit=fit_enc)
            else:
                df[f"{cat}_id"] = -1

        features: List[np.ndarray] = []
        scores: List[float] = []
        scoreset_ids: List[int] = []
        for _, row in df.iterrows():
            feat = self._build_feature_row(row)
            if feat is None:
                continue
            features.append(feat)
            scores.append(float(row["score"]))
            scoreset_ids.append(int(row["scoreset_id"]))

        if features:
            self.feature_names = [f"f_{i}" for i in range(len(features[0]))]

        out_df = pd.DataFrame(
            {
                "score": scores,
                "features": features,
                "scoreset_id": scoreset_ids,
            }
        )
        return out_df

    def forward(self, x: List[Tensor]) -> Tensor:
        if len(x) != 2:
            raise ValueError("Expected [features, scoreset_ids]")
        feats, scoreset_ids = x
        feats_np = feats.detach().cpu().numpy()
        scoreset_ids_np = scoreset_ids.detach().cpu().numpy()
        preds: List[float] = []
        for row, sid in zip(feats_np, scoreset_ids_np):
            sid_val = int(sid)
            model = self.scoreset_models.get(sid_val)
            if model is not None:
                pred = float(model.predict(row.reshape(1, -1))[0])
            else:
                pred = self.scoreset_means.get(sid_val, self.global_mean)
            preds.append(pred)
        return torch.tensor(preds, device=feats.device, dtype=torch.float32)

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,
        optimizer: Optional[Optimizer],
        params: Dict[str, Any],
    ) -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost is required for ManyBoostCompleteModel but is not installed.")

        feature_rows: List[np.ndarray] = []
        scoreset_ids: List[int] = []
        targets: List[float] = []
        for x_row, y in zip(dataset.x, dataset.y):
            feature_rows.append(x_row[0].cpu().numpy())
            scoreset_ids.append(int(x_row[1].item()))
            targets.append(float(y.item()))

        if not feature_rows:
            return

        X_feats = np.stack(feature_rows).astype(np.float32)
        scoreset_ids = np.array(scoreset_ids, dtype=int)
        y_all = np.array(targets, dtype=np.float32)

        self.global_mean = float(y_all.mean()) if len(y_all) else 0.0
        self.scoreset_means = {}
        self.scoreset_models = {}

        for sid in tqdm(np.unique(scoreset_ids), desc="Training final XGB per scoreset"):
            mask = scoreset_ids == sid
            X_sid = X_feats[mask]
            y_sid = y_all[mask]
            self.scoreset_means[sid] = float(y_sid.mean())
            if len(y_sid) < self.min_local:
                continue

            model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                min_child_weight=self.min_child_weight,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                n_jobs=self.n_jobs,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
            )
            model.fit(X_sid, y_sid, verbose=False)
            self.scoreset_models[int(sid)] = model

    def save(self, file_path: str) -> None:
        import pickle

        payload = {
            "scoreset_models": self.scoreset_models,
            "scoreset_means": self.scoreset_means,
            "global_mean": self.global_mean,
            "encoders": self.encoders,
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": self.n_jobs,
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(file_path: str) -> "ManyBoostCompleteModel":
        import pickle

        with open(file_path, "rb") as f:
            payload = pickle.load(f)
        model = ManyBoostCompleteModel(params={})
        model.scoreset_models = payload.get("scoreset_models", {})
        model.scoreset_means = payload.get("scoreset_means", {})
        model.global_mean = payload.get("global_mean", 0.0)
        model.encoders = payload.get("encoders", {})
        model.feature_names = payload.get("feature_names", [])
        model.n_estimators = payload.get("n_estimators", model.n_estimators)
        model.max_depth = payload.get("max_depth", model.max_depth)
        model.learning_rate = payload.get("learning_rate", model.learning_rate)
        model.subsample = payload.get("subsample", model.subsample)
        model.colsample = payload.get("colsample_bytree", model.colsample)
        model.min_child_weight = payload.get("min_child_weight", model.min_child_weight)
        model.reg_alpha = payload.get("reg_alpha", model.reg_alpha)
        model.reg_lambda = payload.get("reg_lambda", model.reg_lambda)
        model.n_jobs = payload.get("n_jobs", model.n_jobs)
        return model

    def prepare_inference(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        df = data.copy()
        if not self.encoders:
            raise RuntimeError("Encoders not fitted. Run transform/train first.")
        df["scoreset_id"] = self._encode_cat(df["scoreset"], "scoreset", fit=False)
        for cat in ["ensp", "ref_long", "alt_long", "biotype"]:
            if cat in df.columns:
                df[f"{cat}_id"] = self._encode_cat(df[cat], cat, fit=False)
            else:
                df[f"{cat}_id"] = -1

        features: List[np.ndarray] = []
        scoreset_ids: List[int] = []
        accessions: List[Any] = []
        for idx, row in df.iterrows():
            feat = self._build_feature_row(row)
            if feat is None:
                continue
            features.append(feat)
            scoreset_ids.append(int(row["scoreset_id"]))
            accessions.append(row.get("accession", idx))

        return np.array(features), np.array(scoreset_ids), accessions
