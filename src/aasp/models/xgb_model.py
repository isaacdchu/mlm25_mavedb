from __future__ import annotations
from typing import Dict, Any, List
import pickle
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .model_interface import Model
from .aasp_dataset import AASPDataset
from .data_handler import DataHandler

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - dependency may be optional in some environments
    XGBRegressor = None  # type: ignore[assignment]
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None


class XGBModel(Model):
    """
    Wrapper around XGBoostRegressor for tabular-only features (Feature Set A).
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: Dict[str, Any] = params
        self.regressor: XGBRegressor | None = None
        self.in_features: int | None = None

    def _check_dependency(self) -> None:
        if XGBRegressor is None:
            raise ImportError(
                "XGBoost is required for XGBModel. "
                "Install with `pip install xgboost`."
            ) from _XGB_IMPORT_ERROR

    def _ensure_rel_pos(self, data: pd.DataFrame) -> pd.DataFrame:
        if "rel_pos" in data.columns:
            return data
        if "pos" in data.columns:
            max_pos: float = float(np.nanmax(data["pos"].astype(float)))
            if max_pos > 0:
                data = data.copy()
                data["rel_pos"] = data["pos"].astype(float) / max_pos
        return data

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
        drop_cols = [col for col in ["accession", "ensp"] if col in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols)
        self.in_features = data.shape[1] - 1  # exclude score
        return data

    def _tensor_list_to_numpy(self, features: List[Tensor]) -> List[float]:
        row: List[float] = []
        for tensor in features:
            if tensor.ndim == 0:
                row.append(float(tensor.item()))
            elif tensor.ndim == 1:
                if tensor.numel() == 1:
                    row.append(float(tensor.item()))
                else:
                    row.extend(tensor.cpu().numpy().astype(np.float32).tolist())
            else:
                row.extend(tensor.flatten().tolist())
        return row

    def forward(self, x: List[Tensor]) -> Tensor:
        self._check_dependency()
        if self.regressor is None:
            raise RuntimeError("Regressor is not trained. Call train_loop before forward.")
        if not x:
            raise ValueError("Input feature list is empty.")
        batch_size: int = x[0].shape[0] if x[0].ndim > 0 else 1
        rows: List[List[float]] = []
        for i in range(batch_size):
            sample: List[float] = []
            for tensor in x:
                if tensor.ndim == 0:
                    sample.append(float(tensor.item()))
                elif tensor.ndim == 1:
                    sample.append(float(tensor[i].item()))
                else:
                    sample.extend(tensor[i].flatten().tolist())
            rows.append(sample)
        preds: np.ndarray = self.regressor.predict(np.array(rows, dtype=np.float32))
        device = x[0].device if x else "cpu"
        return torch.tensor(preds, device=device, dtype=torch.float32).unsqueeze(-1)

    def train_loop(
        self,
        dataset: AASPDataset,
        criterion: Module,  # unused but kept for interface consistency
        optimizer: Optimizer,  # unused but kept for interface consistency
        params: Dict[str, Any]
    ) -> None:
        del criterion, optimizer  # not used for tree-based model
        self._check_dependency()
        x_rows: List[List[float]] = [self._tensor_list_to_numpy(sample) for sample in dataset.x]
        y_vals: List[float] = [float(target.item()) for target in dataset.y]
        x_np: np.ndarray = np.asarray(x_rows, dtype=np.float32)
        y_np: np.ndarray = np.asarray(y_vals, dtype=np.float32)
        if self.regressor is None:
            self.regressor = XGBRegressor(**self.params.get("xgb_params", {}))
        fit_kwargs: Dict[str, Any] = self.params.get("fit_params", {})
        self.regressor.fit(x_np, y_np, **fit_kwargs)

    def save(self, file_path: str) -> None:
        self._check_dependency()
        if self.regressor is None:
            raise RuntimeError("Regressor is not trained; nothing to save.")
        state = {
            "params": self.params,
            "in_features": self.in_features,
            "regressor": self.regressor,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(file_path: str) -> XGBModel:
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        model = XGBModel(params=state.get("params", {}))
        model.in_features = state.get("in_features")
        model.regressor = state.get("regressor")
        model.eval()
        return model
