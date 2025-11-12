# from src.aasp.model.models.dumb_model import DumbModel

# import torch
# from torch.utils.data import DataLoader

# from src.aasp.data_handler.aasp_dataset import AASPDataset


# def main():

#     print(">>> building dataset...")
#     dataset = AASPDataset(
#         config_path="config.yaml",
#         fields=["ref_embedding", "alt_embedding"],
#         fuse_mode="distance",
#         embed_metric="cosine"
#     )

#     print(">>> dataset built.")
#     print(f"Dataset length: {len(dataset)}")
#     print(f"Feature tensor shape: {dataset.X.shape}")   # [N, D]
#     print(f"Target tensor shape:  {dataset.y.shape}")   # [N]

#     loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     print(">>> dataloader constructed.")

#     # --- build model ---
#     input_dim = dataset.X.shape[1]
#     model = DumbModel(input_dim=input_dim)
#     print(f"Model created with input_dim = {input_dim}")

#     # --- one batch sanity check ---
#     X, y = next(iter(loader))
#     print("\n--- sample batch ---")
#     print(f"X batch shape: {X.shape}")
#     print(f"y batch shape: {y.shape}")
#     print(f"first 5 y values: {y[:5].tolist()}")

#     # forward pass
#     y_hat = model(X)
#     print(f"y_hat shape: {y_hat.shape}")
#     print(f"first 5 y_hat values: {y_hat[:5].detach().tolist()}")

#     # quick loss
#     loss = torch.nn.functional.mse_loss(y_hat, y)
#     print(f"sample loss: {float(loss):.6f}")

#     print("\n>>> dumb model sanity run complete — no crashes")


# if __name__ == "__main__":
#     main()

# --------------------------------------------------------------------------------------

# training/main.py
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# project imports
from src.aasp.data_handler.aasp_dataset import AASPDataset
from src.aasp.model.models.dumb_model import DumbModel
from src.aasp.model.models.model import BaselineModel          # <-- use the full model
from src.aasp.model.trainer import Trainer


def main():
    print(">>> building dataset...")
    dataset = AASPDataset(
        config_path="config.yaml",
        fields=["ref_embedding", "alt_embedding", "biotype", "consequence", "ref_long", "alt_long", "scoreset"],
        fuse_mode="distance",            # puts distance at X[:,0]
        embed_metric="cosine",
        categorical_config={
            "biotype": "embedding",      # appends ids as a column
            "ref_long": "embedding",
            "alt_long": "embedding",
            "scoreset": "embedding",
            # "consequence": "multi_hot"   # appends multihot columns
        }
    )
    print(">>> dataset built.")
    print(f"Total samples: {len(dataset):,}")
    print(f"Feature tensor shape: {dataset.X.shape}")
    print(f"Target tensor shape:  {dataset.y.shape}")

    # ---------------- Train/Val split ----------------
    val_frac = 0.20
    N = len(dataset)
    val_size = int(N * val_frac)
    train_size = N - val_size
    print(f">>> splitting: train={train_size:,}, val={val_size:,} (val_frac={val_frac:.2f})")

    # Optional reproducibility for the split
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    # ---------------- DataLoaders ----------------
    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(">>> dataloaders constructed.")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # ---------------- Model/Optimizer/Loss ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> using device: {device}")

    input_dim = dataset.X.shape[1]  # 1 when fuse_mode='distance'
    model = BaselineModel(input_dim=input_dim).to(device)
    print(f">>> model: DumbModel(input_dim={input_dim})")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # quick one-batch sanity check before training
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    with torch.no_grad():
        yhat = model(xb)
        test_loss = loss_fn(yhat, yb).item()
    print("--- sanity check ---")
    print(f"first train batch X shape: {xb.shape}, y shape: {yb.shape}")
    print(f"forward OK, loss={test_loss:.4f}")
    print("--------------------")

    # ---------------- Train ----------------
    # ensure output dir exists
    Path("output").mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,  # tweak as needed
        save_path="output/baseline_model_best.pth",
        device=device
    )
    print(">>> starting training ...")
    trainer.run()
    print(">>> training finished. Best model saved to output/baseline_model_best.pth")


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------------------------

# # training/main.py
# import torch
# from torch.utils.data import DataLoader
# from pathlib import Path

# from src.aasp.data_handler.aasp_dataset import AASPDataset
# from src.aasp.model.models.model import BaselineModel          # <-- use the full model
# from src.aasp.model.trainer import Trainer

# def main():
#     print(">>> building dataset...")
#     fields = ["ref_embedding", "alt_embedding", "biotype", "consequence",
#               "ref_long", "alt_long", "scoreset"]

#     # produce distance + pack categorical columns so Trainer can slice them
#     dataset = AASPDataset(
#         config_path="config.yaml",
#         fields=fields,
#         fuse_mode="distance",          # X[:,0] = distance
#         embed_metric="cosine",
#         categorical_config={
#             "biotype": "embedding",
#             "ref_long": "embedding",
#             "alt_long": "embedding",
#             "scoreset": "embedding",
#             # "consequence": "multi_hot",
#         }
#     )
#     print(">>> dataset built.")
#     print(f"Total samples: {len(dataset):,}")
#     print("Feature tensor shape:", dataset.X.shape)
#     print("Target tensor shape: ", dataset.y.shape)

#     # split (simple 80/20)
#     N = len(dataset)
#     val_frac = 0.20
#     n_val = int(N * val_frac)
#     n_train = N - n_val
#     print(f">>> splitting: train={n_train:,}, val={n_val:,} (val_frac={val_frac:.2f})")

#     g = torch.Generator().manual_seed(0)
#     train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)

#     train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
#     val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, num_workers=0)
#     print(">>> dataloaders constructed.")
#     print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f">>> using device: {device}")

#     # -------- build BaselineModel ----------
#     # input to the MLP head is ONLY the float part we pass as X (distance) → dim = 1
#     input_dim = 1

#     # categorical embedding dims (choose small dims to start)
#     cat_names = [n for n in ["biotype", "ref_long", "alt_long", "scoreset"] if n in dataset.vocabs]
#     cat_dims = {name: (len(dataset.vocabs[name]), 4) for name in cat_names}

#     # multi-hot dimensions
#     multi_hot_dims = {}
#     if "consequence" in dataset.vocabs:
#         multi_hot_dims["consequence"] = len(dataset.vocabs["consequence"])

#     model = BaselineModel(
#         input_dim=input_dim,
#         cat_dims=cat_dims,
#         multi_hot_dims=multi_hot_dims,
#         hidden_dims=(64, 16),
#         dropout_rates=(0.2, 0.1),
#     ).to(device)

#     print(">>> model:", model.__class__.__name__)
#     print(f"cat_dims: { {k: v[0] for k,v in cat_dims.items()} }  (embedding dim=4)")
#     if multi_hot_dims:
#         print(f"multi_hot_dims: {multi_hot_dims}")

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     loss_fn   = torch.nn.MSELoss()

#     # quick sanity forward
#     xb, yb = next(iter(train_loader))
#     xb, yb = xb.to(device), yb.to(device)
#     distance   = xb[:, 0:1]
#     biotype    = xb[:, 1].long()
#     ref_aa     = xb[:, 2].long()
#     alt_aa     = xb[:, 3].long()
#     scoreset   = xb[:, 4].long()
#     consequence = xb[:, 5:]

#     with torch.no_grad():
#         yh = model(
#             distance,
#             biotype=biotype if "biotype" in cat_dims else None,
#             ref_aa=ref_aa if "ref_long" in cat_dims else None,     # if you used 'ref_long' as name
#             alt_aa=alt_aa if "alt_long" in cat_dims else None,
#             scoreset=scoreset if "scoreset" in cat_dims else None,
#             consequence=consequence if "consequence" in multi_hot_dims else None,
#         )
#         test_loss = loss_fn(yh, yb)
#         print("--- sanity check ---")
#         print(f"first train batch X shape: {xb.shape}, y shape: {yb.shape}")
#         print(f"forward OK, loss={float(test_loss):.4f}")
#         print("--------------------")

#     # -------- train ----------
#     print(">>> starting training ...")
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         loss_fn=loss_fn,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=5,
#         save_path=str(Path("output") / "baseline_model_best.pth"),
#         device=device,
#     )
#     trainer.run()
#     print(">>> training complete.")

# if __name__ == "__main__":
#     main()
