from __future__ import annotations
import argparse
import inspect
import os
from pathlib import Path
from typing import List, Dict, Type, Tuple, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.aasp import (
    AASPDataset,
    DataHandler,
    Trainer
)
import src.aasp.models as model_registry


def _is_git_lfs_pointer(file_path: str) -> bool:
    """
    Detects Git LFS pointer files to avoid attempting to unpickle them.
    """
    try:
        with open(file_path, "rb") as f:
            snippet = f.read(200)
        try:
            decoded = snippet.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
        first_line = decoded.splitlines()[0] if decoded else ""
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _normalize_embedding_column(col: pd.Series) -> pd.Series:
    """
    Ensure embedding columns are numpy float arrays (no tensors/objects).
    """
    def _convert(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy().astype(np.float32)
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.asarray(val, dtype=np.float32)
        return val
    return col.apply(_convert)


def _ensure_output_dirs() -> None:
    Path("output/models").mkdir(parents=True, exist_ok=True)
    Path("output/submissions").mkdir(parents=True, exist_ok=True)


def _load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a dataframe from CSV or pickle.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(f"Pickle at {path} did not contain a pandas DataFrame (found {type(obj)}).")
    raise ValueError(f"Unsupported data file extension for {path}; use .csv or .pkl.")


def _load_auxiliary_maps() -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Load optional embedding and sequence length maps if present.
    Returns a tuple of (embedding_map, sequence_length_map).
    """
    embedding_map: Dict[str, np.ndarray] = {}
    seq_len_map: Dict[str, int] = {}
    emb_path = "data/ensp_embeddings_map.pkl"
    seq_path = "data/ensp_sequence_map.pkl"
    if os.path.exists(emb_path):
        if _is_git_lfs_pointer(emb_path):
            print(f"Warning: {emb_path} is a Git LFS pointer; embeddings unavailable. Run 'git lfs fetch' to retrieve.")
        else:
            try:
                import pickle
                with open(emb_path, "rb") as f:
                    raw_map = pickle.load(f)
                    embedding_map = {k: np.asarray(v, dtype=np.float32) for k, v in raw_map.items()}
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Warning: failed to load {emb_path}: {exc}")
    else:
        print(f"Warning: {emb_path} not found; embeddings will be missing.")

    if os.path.exists(seq_path):
        if _is_git_lfs_pointer(seq_path):
            print(f"Warning: {seq_path} is a Git LFS pointer; sequence lengths unavailable. Run 'git lfs fetch' to retrieve.")
        else:
            try:
                import pickle
                with open(seq_path, "rb") as f:
                    seq_map: Dict[str, str] = pickle.load(f)
                    seq_len_map = {k: len(v) for k, v in seq_map.items()}
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Warning: failed to load {seq_path}: {exc}")
    else:
        print(f"Warning: {seq_path} not found; sequence lengths will be missing.")
    return embedding_map, seq_len_map


def _attach_metadata(
    df: pd.DataFrame,
    emb_map: Dict[str, np.ndarray],
    seq_len_map: Dict[str, int],
    key_col: str = "ensp",
    warn: bool = True
) -> pd.DataFrame:
    """
    Join sequence_length and embeddings onto the dataframe using the provided maps.
    Missing embeddings get zero vectors (if dimension is known) and emit a warning.
    """
    enriched: pd.DataFrame = df.copy()
    merge_key: Optional[str] = key_col if key_col in enriched.columns else ("accession" if "accession" in enriched.columns else None)

    if merge_key is not None:
        if seq_len_map:
            enriched["sequence_length"] = enriched[merge_key].map(seq_len_map)
        elif "sequence_length" not in enriched.columns:
            enriched["sequence_length"] = np.nan

        emb_dim: Optional[int] = None
        if emb_map:
            sample = next(iter(emb_map.values()))
            emb_dim = int(sample.shape[0])

        missing = 0

        def _fetch_embedding(key: str):
            nonlocal missing
            emb = emb_map.get(key)
            if emb is None:
                missing += 1
                if emb_dim is not None:
                    return np.zeros(emb_dim, dtype=np.float32)
                return np.nan
            return emb

        if emb_map:
            enriched["ref_embedding"] = enriched[merge_key].apply(_fetch_embedding)
            enriched["alt_embedding"] = enriched[merge_key].apply(_fetch_embedding)
            if warn and missing:
                print(f"Warning: {missing} rows missing embeddings for key '{merge_key}'; filled with zeros.")
        elif warn:
            print("Warning: embedding map empty; ref_embedding/alt_embedding set to NaN.")
            enriched["ref_embedding"] = np.nan
            enriched["alt_embedding"] = np.nan

    enriched = DataHandler.add_rel_pos(enriched, sequence_length_map=seq_len_map)
    return enriched


def _get_available_models() -> Dict[str, Type[model_registry.Model]]:
    models: Dict[str, Type[model_registry.Model]] = {}
    for name, cls in vars(model_registry).items():
        if not inspect.isclass(cls):
            continue
        if not issubclass(cls, model_registry.Model):
            continue
        if cls is model_registry.Model:
            continue
        if getattr(cls, "__abstractmethods__", set()):
            continue
        if name == "ExampleModel":
            continue
        models[name] = cls  # type: ignore[assignment]
    return models


def _train_and_eval_single(
    model_cls: Type[model_registry.Model],
    data: pd.DataFrame,
    device: torch.device,
    epochs: int,
    batch_size: int
) -> float:
    model_params: Dict[str, Any] = {}
    if model_cls.__name__ == "XGBModel":
        model_params = {
            "xgb_params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "reg:squarederror",
                "n_jobs": max(1, os.cpu_count() or 1),
            },
            "fit_params": {
                "verbose": False,
            },
        }

    model = model_cls(params=model_params)
    transformed = model.transform(data.copy())
    if "score" not in transformed.columns:
        raise ValueError("Transformed data missing 'score' column.")

    if "scoreset_id" in transformed.columns:
        groups = transformed["scoreset_id"]
    elif "scoreset" in transformed.columns:
        groups = transformed["scoreset"]
    else:
        groups = None

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(transformed, groups=groups))
        train_df = transformed.iloc[train_idx].reset_index(drop=True)
        val_df = transformed.iloc[val_idx].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(transformed, test_size=0.2, random_state=42)

    train_dataset = AASPDataset(train_df, device=str(device))
    val_dataset = AASPDataset(val_df, device=str(device))

    trainer = Trainer(model)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    params = {"batch_size": batch_size, "num_epochs": epochs}

    # Create a safe optimizer even for models without parameters.
    params_list = [p for p in model.parameters() if p.requires_grad]
    if not params_list:
        params_list = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
    optimizer = torch.optim.Adam(params_list, lr=1e-3)

    trainer.train(train_dataset, criterion, optimizer, params)

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            features = [tensor.to(device=device, dtype=torch.float32) for tensor in x_batch]
            targets = y_batch.to(device=device, dtype=torch.float32)
            preds = model.forward(features)
            mse_sum += float(F.mse_loss(preds, targets, reduction="sum").cpu())
            count += targets.numel()
    return mse_sum / max(count, 1)


def _train_full(
    model_cls: Type[model_registry.Model],
    data: pd.DataFrame,
    device: torch.device,
    epochs: int,
    batch_size: int
) -> model_registry.Model:
    model_params: Dict[str, Any] = {}
    if model_cls.__name__ == "XGBModel":
        model_params = {
            "xgb_params": {
                "n_estimators": 800,
                "learning_rate": 0.05,
                "max_depth": 7,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "reg:squarederror",
                "n_jobs": max(1, os.cpu_count() or 1),
            },
            "fit_params": {
                "verbose": False,
            },
        }
    model = model_cls(params=model_params)
    transformed = model.transform(data.copy())
    dataset = AASPDataset(transformed, device=str(device))
    trainer = Trainer(model)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    params_list = [p for p in model.parameters() if p.requires_grad]
    if not params_list:
        params_list = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
    optimizer = torch.optim.Adam(params_list, lr=1e-3)
    trainer.train(dataset, criterion, optimizer, {"batch_size": batch_size, "num_epochs": epochs})
    model.eval()
    return model


def _predict_dataframe(
    model: model_registry.Model,
    df: pd.DataFrame,
    device: torch.device
) -> np.ndarray:
    if "score" not in df.columns:
        df = df.copy()
        df["score"] = 0.0
    transformed = model.transform(df.copy())
    dataset = AASPDataset(transformed, device=str(device))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    preds_list: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            features = [tensor.to(device=device, dtype=torch.float32) for tensor in x_batch]
            outputs = model.forward(features)
            preds_list.append(outputs.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds_list, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or all AASP models on the training data.")
    parser.add_argument("--model", type=str, help="Name of a single model class to run.")
    parser.add_argument("--all-models", action="store_true", help="Run all registered non-abstract models.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--train-path", type=str, default="data/train/combined_train_data.pkl", help="Path to the training data (.pkl or .csv).")
    parser.add_argument("--test-path", type=str, default="data/test/combined_test_data.pkl", help="Path to the test data (.pkl or .csv, optional).")
    parser.add_argument("--save-model", action="store_true", help="Save trained model to output/models/.")
    parser.add_argument("--predict", action="store_true", help="Load saved model and generate submission CSV.")
    parser.add_argument("--submission-name", type=str, default=None, help="Optional submission filename override.")
    args = parser.parse_args()

    available_models = _get_available_models()
    if args.model:
        selected = {args.model: available_models.get(args.model)}
        if selected[args.model] is None:
            raise ValueError(f"Model '{args.model}' not found. Available: {sorted(available_models.keys())}")
        models_to_run = selected  # type: ignore[assignment]
    elif args.all_models:
        models_to_run = available_models
    else:
        raise ValueError("Specify --model <Name> or --all-models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_train = Path("data/train/combined_train_data.pkl")
    default_test = Path("data/test/combined_test_data.pkl")
    use_combined = default_train.exists() and default_test.exists()

    if use_combined:
        mode_banner = "Predicting" if args.predict else "Training"
        target = list(models_to_run.keys())[0] if len(models_to_run) == 1 else "models"
        print(f"{mode_banner} {target} on combined dataset pickles.")
        train_df = pd.read_pickle(default_train)
        if "consequences" in train_df.columns:
            train_df = DataHandler.multi_hot_encode(train_df, columns=["consequences"])
        if "ref_embedding" in train_df.columns:
            train_df["ref_embedding"] = _normalize_embedding_column(train_df["ref_embedding"])
        if "alt_embedding" in train_df.columns:
            train_df["alt_embedding"] = _normalize_embedding_column(train_df["alt_embedding"])
        enriched_df = DataHandler.add_rel_pos(train_df)
        if default_test.exists():
            test_df = pd.read_pickle(default_test)
            if "consequences" in test_df.columns:
                test_df = DataHandler.multi_hot_encode(test_df, columns=["consequences"])
            if "ref_embedding" in test_df.columns:
                test_df["ref_embedding"] = _normalize_embedding_column(test_df["ref_embedding"])
            if "alt_embedding" in test_df.columns:
                test_df["alt_embedding"] = _normalize_embedding_column(test_df["alt_embedding"])
            _ = DataHandler.add_rel_pos(test_df)
    else:
        print("Using raw CSV + auxiliary maps.")
        emb_map, seq_len_map = _load_auxiliary_maps()
        raw_df = _load_dataframe(args.train_path)
        enriched_df = _attach_metadata(raw_df, emb_map=emb_map, seq_len_map=seq_len_map)

        if args.test_path:
            try:
                raw_test_df = _load_dataframe(args.test_path)
                _attach_metadata(raw_test_df, emb_map=emb_map, seq_len_map=seq_len_map)
            except FileNotFoundError:
                print(f"Warning: test data not found at {args.test_path}")

    _ensure_output_dirs()

    if args.predict:
        if args.model is None:
            raise ValueError("Prediction requires --model to be specified.")
        model_name = args.model
        model_cls = available_models.get(model_name)
        if model_cls is None:
            raise ValueError(f"Model '{model_name}' not found.")
        model_path = Path("output/models") / (f"{model_name}.pkl" if model_name == "XGBModel" else f"{model_name}.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found at {model_path}")
        model = model_cls.load(str(model_path))
        test_df = pd.read_pickle(default_test if use_combined else args.test_path)
        if "consequences" in test_df.columns:
            test_df = DataHandler.multi_hot_encode(test_df, columns=["consequences"])
        for col in ["ref_embedding", "alt_embedding"]:
            if col in test_df.columns:
                test_df[col] = _normalize_embedding_column(test_df[col])
        test_df = DataHandler.add_rel_pos(test_df)
        id_col = "id" if "id" in test_df.columns else "accession" if "accession" in test_df.columns else test_df.columns[0]
        preds = _predict_dataframe(model, test_df, device=device)
        preds = np.asarray(preds, dtype=np.float32)
        if np.any(~np.isfinite(preds)):
            finite_mask = np.isfinite(preds)
            fill = float(np.nanmean(preds[finite_mask])) if finite_mask.any() else 0.0
            preds[~finite_mask] = fill
        timestamp = args.submission_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_path = Path("output/submissions") / f"{timestamp}_{model_name}.csv"
        pd.DataFrame({id_col: test_df[id_col], "prediction": preds}).to_csv(sub_path, index=False)
        print(f"Predicting with {model_name} \u2192 {sub_path}")
        try:
            from submission_validator import validate  # type: ignore
            ok, msg = validate(str(sub_path))
            print(f"Local submission validation: {'PASS' if ok else 'FAIL'}")
            if msg:
                print(msg)
        except Exception:
            pass
        return

    for name, cls in models_to_run.items():
        if cls is None:
            continue
        try:
            mse = _train_and_eval_single(
                model_cls=cls,
                data=enriched_df,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            print(f"{name}: validation MSE={mse:.6f}")
            if args.save_model:
                trained_model = _train_full(
                    model_cls=cls,
                    data=enriched_df,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                model_path = Path("output/models") / (f"{name}.pkl" if name == "XGBModel" else f"{name}.pt")
                trained_model.save(str(model_path))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{name}: failed with error: {exc}")


if __name__ == "__main__":
    main()
