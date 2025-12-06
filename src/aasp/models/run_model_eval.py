"""
Utility script to train a given Model implementation and report MSE on a
validation split. Defaults to the ScoresetEnsembleModel to mirror the
per-scoreset approach used in 100_models.ipynb.
"""

from __future__ import annotations
import argparse
import importlib
import pandas as pd
import math
import os
from typing import Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

from .data_handler import DataHandler
from .aasp_dataset import AASPDataset
from .model_interface import Model


def resolve_model(model_path: str, params: dict[str, Any]) -> Model:
    """
    Load a Model subclass from a dotted path like
    'src.aasp.models.scoreset_ensemble_model.ScoresetEnsembleModel'.
    """
    module_path, cls_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(params)


def train_and_eval(
    model: Model,
    dataset: AASPDataset,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
) -> Tuple[float, float, List[Tuple[str, int, float, float]]]:
    """
    Train the model on a train split and evaluate MSE/MAE on the val split.
    """
    # Train (per-scoreset validation is handled internally by the model)
    criterion = nn.MSELoss()
    # Some Model implementations (like ScoresetEnsembleModel) create parameters lazily,
    # so guard against empty parameter lists here.
    params = list(model.parameters())
    optimizer = (
        torch.optim.Adam(params, lr=1e-3)
        if params
        else None  # train_loop may choose to construct its own optimizer
    )
    model.to(device)  # type: ignore[call-arg]
    dataset.device = device  # type: ignore[attr-defined]
    model.train_loop(
        dataset,
        criterion=criterion,
        optimizer=optimizer,
        params={
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "device": device,
        },
    )

    # Evaluate on the val splits stored by the model (per scoreset)
    model.eval()
    overall_preds = []
    overall_targets = []
    per_scoreset_metrics = []
    val_sets = getattr(model, "val_sets", {})
    if not val_sets:
        return math.nan, math.nan, []
    with torch.no_grad():
        for sid_key, (feats, labels) in val_sets.items():
            feats = feats.to(device)
            labels = labels.to(device)
            preds = model([feats, torch.tensor([int(sid_key)] * len(feats), device=device, dtype=torch.long)])
            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()
            mse_sid = torch.mean((preds_cpu - labels_cpu) ** 2).item()
            mae_sid = torch.mean(torch.abs(preds_cpu - labels_cpu)).item()
            per_scoreset_metrics.append((sid_key, len(labels_cpu), mse_sid, mae_sid))
            overall_preds.append(preds_cpu)
            overall_targets.append(labels_cpu)
    preds_t = torch.cat(overall_preds)
    targets_t = torch.cat(overall_targets)
    mse = torch.mean((preds_t - targets_t) ** 2).item()
    mae = torch.mean(torch.abs(preds_t - targets_t)).item()
    return mse, mae, per_scoreset_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Model implementation and report validation MSE."
    )
    parser.add_argument(
        "--model",
        default="src.aasp.models.scoreset_ensemble_model.ScoresetEnsembleModel",
        help="Dotted path to Model subclass (default: ScoresetEnsembleModel).",
    )
    parser.add_argument(
        "--data",
        default="data/train/combined_train_data.pkl",
        help="Path to training data pickle (default: data/train/combined_train_data.pkl).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Device to use ("cpu", "cuda", or "mps").',
    )
    parser.add_argument(
        "--test-data",
        default="data/test/combined_test_data.pkl",
        help="Path to test data pickle for generating predictions (default: data/test/combined_test_data.pkl).",
    )
    parser.add_argument(
        "--output",
        default="output.csv",
        help="Path to write test predictions CSV (default: output.csv).",
    )
    parser.add_argument(
        "--plot-worst",
        type=int,
        default=5,
        help="Number of worst scoresets (by val MSE) to plot y_true vs y_pred (default: 5).",
    )
    parser.add_argument(
        "--plot-dir",
        default="val_plots",
        help="Directory to save validation plots for worst scoresets (default: val_plots).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading data from {args.data} ...")
    raw_df = DataHandler.load_data(args.data)

    print(f"Instantiating model: {args.model}")
    model = resolve_model(args.model, params={"device": args.device})

    print("Transforming data ...")
    transformed = model.transform(raw_df)
    dataset = AASPDataset(transformed, device=args.device, transform=None)
    print(f"Dataset size: {len(dataset)} samples")

    mse, mae, per_scoreset = train_and_eval(
        model=model,
        dataset=dataset,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    print(f"\nValidation metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("\nPer-scoreset validation metrics (sid, n, MSE, MAE):")
    for sid_key, n, mse_sid, mae_sid in sorted(per_scoreset, key=lambda x: x[0]):
        print(f"  {sid_key}: n={n}, MSE={mse_sid:.4f}, MAE={mae_sid:.4f}")

    # Plot worst scoresets by MSE
    if per_scoreset:
        worst = sorted(per_scoreset, key=lambda x: x[2], reverse=True)[: max(0, args.plot_worst)]
        os.makedirs(args.plot_dir, exist_ok=True)
        val_sets = getattr(model, "val_sets", {})
        for sid_key, n, mse_sid, mae_sid in worst:
            if sid_key not in val_sets:
                continue
            feats, labels = val_sets[sid_key]
            feats = feats.to(args.device)
            labels = labels.to(args.device)
            sid_tensor = torch.tensor([int(sid_key)] * len(feats), device=args.device, dtype=torch.long)
            with torch.no_grad():
                preds = model([feats, sid_tensor]).cpu().numpy()
            labels_np = labels.cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.scatter(labels_np, preds, alpha=0.6)
            min_v = min(labels_np.min(), preds.min())
            max_v = max(labels_np.max(), preds.max())
            plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
            plt.xlabel("True score")
            plt.ylabel("Predicted score")
            plt.title(f"Scoreset {sid_key} (n={n}, MSE={mse_sid:.2f}, MAE={mae_sid:.2f})")
            plt.tight_layout()
            out_path = os.path.join(args.plot_dir, f"scoreset_{sid_key}_val.png")
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved plot for scoreset {sid_key} to {out_path}")

    # Generate predictions on test data
    print(f"\nLoading test data from {args.test_data} ...")
    test_df = DataHandler.load_data(args.test_data)
    print("Preparing features for inference ...")
    test_features, test_scoreset_ids, accessions = model.prepare_inference(test_df)
    if len(test_features) == 0:
        print("No valid test samples to predict.")
        return

    model.eval()
    feats_tensor = torch.tensor(test_features, dtype=torch.float32, device=args.device)
    sid_tensor = torch.tensor(test_scoreset_ids, dtype=torch.long, device=args.device)
    with torch.no_grad():
        preds = model([feats_tensor, sid_tensor]).cpu().numpy()

    output_df = pd.DataFrame(
        {
            "accession": accessions,
            "score": preds,
        }
    )
    output_df.to_csv(args.output, index=False)
    print(f"\nSaved test predictions to {args.output}")


if __name__ == "__main__":
    main()
