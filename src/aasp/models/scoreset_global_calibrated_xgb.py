"""
ScoresetGlobalCalibratedXGBModel
--------------------------------
- Global XGBoost regressor trained on all scoresets with per-scoreset target
  standardization.
- Predictions are de-standardized and then calibrated via a per-scoreset
  Ridge regression to correct residual bias (especially for tiny sets).
- Integrates with the existing training/eval pipeline via transform,
  train_loop, forward, prepare_inference, and val_sets.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

import torch
from torch import nn, Tensor

from .aasp_dataset import AASPDataset
from .model_interface import Model
from tqdm import tqdm
try:
    from xgboost.callback import TrainingCallback
except ImportError:
    TrainingCallback = None  # type: ignore


class ScoresetGlobalCalibratedXGBModel(Model):
    """
    ScoresetGlobalCalibratedXGBModel

    - Uses a single global XGBoost regressor trained on all scoresets.
    - Target is standardized within each scoreset:
          z = (score - mean_s) / std_s
      to avoid high-magnitude scoresets dominating the loss.
    - At prediction time, standardized predictions are mapped back to the
      original scale via stored (mean_s, std_s) per scoreset.
    - A per-scoreset Ridge calibration is then applied:
          y_true ≈ a_s * y_pred + b_s
      which corrects residual bias for each scoreset, especially small,
      high-variance ones.
    - Integrates with the existing train/eval pipeline via:
          - transform(...)
          - train_loop(...)
          - prepare_inference(...)
          - forward(...)
          - val_sets tracking.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.device: str = params.get("device", "cpu")
        default_xgb = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        user_xgb = params.get("xgb_params", {})
        self.xgb_params: Dict[str, Any] = {**default_xgb, **user_xgb}

        self.min_calib_points: int = int(params.get("min_calib_points", 5))

        self.model: Optional[xgb.XGBRegressor] = None
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.scoreset_stats: Dict[int, Tuple[float, float]] = {}
        self.calibrators: Dict[int, Ridge] = {}
        self.val_sets: Dict[str, Tuple[Tensor, Tensor]] = {}
        self._train_df: Optional[pd.DataFrame] = None

        self.cat_columns: List[str] = ["accession", "scoreset", "ensp", "ref_long", "alt_long", "biotype"]

    # ------------------------------------------------------------------
    def _encode_cat(self, value: str, col: str) -> int:
        mapping = self.cat_maps.get(col, {})
        return mapping.get(value, -1)

    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds feature vectors using embedding diffs + categorical IDs + position.
        Returns:
            X: float32 array of shape (N, D)
            y: float32 array of raw scores (if available)
            scoresets: int array of scoreset ids (encoded)
        """
        feature_rows: List[np.ndarray] = []
        scores: List[float] = []
        scoresets: List[int] = []

        for _, row in df.iterrows():
            try:
                ref_emb = np.array(row["ref_embedding"], dtype=np.float32)
                alt_emb = np.array(row["alt_embedding"], dtype=np.float32)
            except Exception:
                continue
            if ref_emb.shape != alt_emb.shape:
                continue
            diff = alt_emb - ref_emb
            abs_diff = np.abs(diff)

            cat_ids: List[float] = []
            for col in self.cat_columns:
                val = str(row.get(col, "UNKNOWN"))
                cat_ids.append(float(self._encode_cat(val, col)))

            pos_val = row.get("pos", -1.0)
            if pd.isna(pos_val):
                pos_val = -1.0
            pos_feat = np.array([float(pos_val)], dtype=np.float32)

            feat_vec = np.concatenate([diff, abs_diff, np.array(cat_ids, dtype=np.float32), pos_feat])
            feature_rows.append(feat_vec)

            scores.append(float(row.get("score", 0.0)))
            try:
                sid = int(row.get("scoreset_id", -1))
            except Exception:
                sid = -1
            scoresets.append(sid)

        if not feature_rows:
            return np.empty((0, 1), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        X = np.stack(feature_rows).astype(np.float32)
        y = np.array(scores, dtype=np.float32)
        sid_arr = np.array(scoresets, dtype=np.int64)
        return X, y, sid_arr

    # ------------------------------------------------------------------
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        for col in self.cat_columns:
            df[col] = df[col].astype(str)
            uniques = df[col].dropna().unique()
            self.cat_maps[col] = {v: i for i, v in enumerate(uniques)}

        df["scoreset_id"] = df["scoreset"].map(self.cat_maps.get("scoreset", {})).fillna(-1).astype(int)

        grouped = df.groupby("scoreset_id")["score"]
        self.scoreset_stats = {}
        for sid, series in grouped:
            mu = float(series.mean())
            sigma = float(series.std())
            if sigma == 0.0:
                sigma = 1.0
            self.scoreset_stats[int(sid)] = (mu, sigma)

        X, y_raw, sid_arr = self._build_features(df)
        score_std = []
        for i, sid in enumerate(sid_arr):
            mu, sigma = self.scoreset_stats.get(int(sid), (0.0, 1.0))
            score_std.append((y_raw[i] - mu) / sigma)
        score_std = np.array(score_std, dtype=np.float32)

        processed = pd.DataFrame(
            {
                "score": y_raw,
                "score_std": score_std,
                "scoreset_id": sid_arr.astype(int),
                "features": list(X),
            }
        )
        self._train_df = processed
        return processed

    # ------------------------------------------------------------------
    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        params: Dict[str, Any],
    ) -> None:
        # Reconstruct the DataFrame stored in the dataset (transform output)
        if self._train_df is None:
            raise RuntimeError("transform must be called before train_loop.")
        transformed_df = self._train_df

        self.val_sets = {}
        train_rows: List[pd.DataFrame] = []

        for sid_int, group in tqdm(transformed_df.groupby("scoreset_id"), desc="Preparing per-scoreset splits"):
            rows = group.sample(frac=1.0, random_state=42)  # shuffle
            split_idx = max(1, int(0.2 * len(rows)))
            val_part = rows.iloc[:split_idx]
            train_part = rows.iloc[split_idx:] if split_idx < len(rows) else rows
            if train_part.empty:
                train_part = rows

            train_rows.append(train_part)

            # Build validation tensors (features + original score)
            feats_val = np.stack(val_part["features"].to_list()).astype(np.float32)
            labels_val = val_part["score"].to_numpy(dtype=np.float32)
            feats_tensor = torch.tensor(feats_val, dtype=torch.float32, device=self.device)
            labels_tensor = torch.tensor(labels_val, dtype=torch.float32, device=self.device)
            self.val_sets[str(sid_int)] = (feats_tensor, labels_tensor)

        if not train_rows:
            raise RuntimeError("No training rows after grouping by scoreset.")

        train_df = pd.concat(train_rows, ignore_index=True)
        X_train = np.stack(train_df["features"].to_list()).astype(np.float32)
        sid_arr = train_df["scoreset_id"].to_numpy(dtype=np.int64)
        y_train_std = train_df["score_std"].to_numpy(dtype=np.float32)

        self.model = xgb.XGBRegressor(**self.xgb_params)

        progress_bar = tqdm(total=self.xgb_params.get("n_estimators", 0), desc="Training global XGB")
        if TrainingCallback is not None and self.xgb_params.get("n_estimators", 0) > 0:
            class _TQDMCallback(TrainingCallback):  # type: ignore[misc]
                def __init__(self, bar: tqdm) -> None:
                    self.bar = bar

                def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Any]) -> bool:
                    self.bar.update(1)
                    return False

                def after_training(self, model: xgb.Booster) -> None:
                    self.bar.close()

            try:
                self.model.fit(X_train, y_train_std, callbacks=[_TQDMCallback(progress_bar)])
            except TypeError:
                progress_bar.close()
                self.model.fit(X_train, y_train_std)
        else:
            progress_bar.close()
            self.model.fit(X_train, y_train_std)

        y_pred_std = self.model.predict(X_train)
        y_pred_orig = np.zeros_like(y_pred_std, dtype=np.float32)
        y_true_orig = train_df["score"].to_numpy(dtype=np.float32)

        for i, sid in enumerate(sid_arr):
            mu, sigma = self.scoreset_stats.get(int(sid), (0.0, 1.0))
            y_pred_orig[i] = y_pred_std[i] * sigma + mu

        self.calibrators = {}
        for sid in tqdm(np.unique(sid_arr), desc="Fitting per-scoreset calibrators"):
            mask = sid_arr == sid
            y_true_s = y_true_orig[mask]
            y_pred_s = y_pred_orig[mask]
            if len(y_true_s) >= self.min_calib_points:
                reg = Ridge(alpha=1.0)
                reg.fit(y_pred_s.reshape(-1, 1), y_true_s)
                self.calibrators[int(sid)] = reg

        # auto-save trained model
        try:
            import os
            os.makedirs("models/saved_models", exist_ok=True)
            self.save("models/saved_models/global_calibrated_xgb")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def forward(self, inputs: List[Tensor]) -> Tensor:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        feats, sid_tensor = inputs
        X = feats.detach().cpu().numpy()
        sid_arr = sid_tensor.detach().cpu().numpy()
        y_pred_std = self.model.predict(X)

        y_pred_orig = np.zeros_like(y_pred_std, dtype=np.float32)
        for i, sid in enumerate(sid_arr):
            mu, sigma = self.scoreset_stats.get(int(sid), (0.0, 1.0))
            y_pred_orig[i] = y_pred_std[i] * sigma + mu

        y_final = np.zeros_like(y_pred_orig, dtype=np.float32)
        for i, sid in enumerate(sid_arr):
            reg = self.calibrators.get(int(sid))
            if reg is not None:
                y_final[i] = reg.predict([[float(y_pred_orig[i])]])[0]
            else:
                y_final[i] = y_pred_orig[i]

        # Auto-save after training
        try:
            self.save("models/saved_models/global_calibrated_xgb_full.pt")
        except Exception:
            pass
        return torch.tensor(y_final, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def prepare_inference(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        inf_df = df.copy()
        for col in self.cat_columns:
            inf_df[col] = inf_df[col].astype(str)
        if "scoreset" in inf_df.columns:
            inf_df["scoreset_id"] = inf_df["scoreset"].map(self.cat_maps.get("scoreset", {})).fillna(-1).astype(int)
        else:
            inf_df["scoreset_id"] = -1

        features, _, scoresets = self._build_features(inf_df)
        accessions = inf_df["accession"].astype(str).tolist()
        return features, scoresets.astype(np.int64), accessions

    # ------------------------------------------------------------------
    def save(self, file_path: str) -> None:
        import pickle

        if self.model is None:
            raise RuntimeError("Model not trained; nothing to save.")

        payload = {
            "model": self.model.get_booster().save_raw("json"),
            "xgb_params": self.xgb_params,
            "cat_maps": self.cat_maps,
            "scoreset_stats": self.scoreset_stats,
            "calibrators": {sid: (reg.coef_, reg.intercept_) for sid, reg in self.calibrators.items()},
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(file_path: str) -> "ScoresetGlobalCalibratedXGBModel":
        import pickle

        with open(file_path, "rb") as f:
            payload = pickle.load(f)

        model = ScoresetGlobalCalibratedXGBModel(params={"xgb_params": payload.get("xgb_params", {})})
        booster = xgb.Booster()
        booster.load_model(bytearray(payload["model"]))
        model.model = xgb.XGBRegressor(**model.xgb_params)
        model.model._Booster = booster

        model.cat_maps = payload.get("cat_maps", {})
        model.scoreset_stats = payload.get("scoreset_stats", {})

        calibrators_payload = payload.get("calibrators", {})
        model.calibrators = {}
        for sid, (coef, intercept) in calibrators_payload.items():
            reg = Ridge(alpha=1.0)
            reg.coef_ = np.array(coef, dtype=np.float64)
            reg.intercept_ = float(intercept)
            reg.n_features_in_ = 1
            model.calibrators[int(sid)] = reg

        return model
