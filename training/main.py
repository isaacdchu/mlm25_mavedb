import torch
from torch.utils.data import DataLoader, random_split
from training.src.aasp.data_handler.data_handler import AASPConfig, AASPDataHandler
from feature_engineering.aasp_dataset import AASPDataset
from training.src.aasp.model.model import BaselineModel

def main(config_path="training/src/aasp/data_handler/config.yaml"):
    # --- Load config and handler ---
    cfg = AASPConfig(config_path)
    handler = AASPDataHandler(cfg)

    # --- Hyperparameters and feature selection from config ---
    hyper = getattr(cfg, "hyperparameters", {})
    batch_size = hyper.get("train_batch_size", 32)
    val_batch_size = hyper.get("val_batch_size", 256)
    num_epochs = hyper.get("num_epochs", 10)
    learning_rate = hyper.get("learning_rate", 0.001)
    hidden_dims = tuple(hyper.get("hidden_dims", [64, 16]))
    dropout_rates = tuple(hyper.get("dropout_rates", [0.2, 0.1]))
    fuse_mode = getattr(cfg, "fuse_mode", "distance")
    embed_metric = getattr(cfg, "embed_metric", "cosine")
    cat_config = getattr(cfg, "categorical_config", {
        "biotype": "embedding", "ref_long": "embedding", "alt_long": "embedding", "consequence": "multi_hot"
    })

    selected_features = [
        k for k, v in getattr(cfg, "features", {}).items()
        if v and k in {"ref_embedding", "alt_embedding", "biotype", "ref_long", "alt_long", "consequence"}
    ]

    # --- Load and split data ---
    records = handler.load_pickle(cfg.file_path)
    val_frac = hyper.get("val_frac", 0.15)
    val_size = int(len(records) * val_frac)
    train_size = len(records) - val_size
    train_records = records[:train_size]
    val_records = records[train_size:]

    # --- Fit vocabularies from train only ---
    vocabs = {}
    for cat in cat_config.keys():
        vocabs[cat] = handler.fit_vocab(train_records, cat)
    if "consequence" not in vocabs and "consequence" in cat_config:
        vocabs["consequence"] = handler.fit_vocab(train_records, "consequence")

    # --- Datasets ---
    train_dataset = AASPDataset(
        config_path=config_path,
        fields=selected_features,
        fuse_mode=fuse_mode,
        embed_metric=embed_metric,
        categorical_config=cat_config
    )
    val_dataset = AASPDataset(
        config_path=config_path,       # reuse config
        fields=selected_features,
        fuse_mode=fuse_mode,
        embed_metric=embed_metric,
        categorical_config=cat_config
    )
    train_dataset.records = train_records
    val_dataset.records = val_records

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # --- Model dimensions ---
    cat_dims = {
        k: (len(vocabs[k]), 4) for k in cat_config if cat_config[k] == "embedding"
    }
    multi_hot_dims = {
        k: len(vocabs[k]) for k in cat_config if cat_config[k] == "multi_hot"
    }
    input_dim = 1 # "distance" by default, or infer from actual shape if needed

    model = BaselineModel(
        input_dim=input_dim,
        cat_dims=cat_dims,
        multi_hot_dims=multi_hot_dims,
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # --- Training and validation ---
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for (X, y) in train_loader:
            # unpack X into features if needed (depends on AASPDataset __getitem__)
            distance = X[:, 0:1]
            biotype = X[:, 1].long()
            ref_aa = X[:, 2].long()
            alt_aa = X[:, 3].long()
            consequence = X[:, 4:]

            optimizer.zero_grad()
            y_hat = model(
                distance,
                biotype=biotype,
                ref_aa=ref_aa,
                alt_aa=alt_aa,
                consequence=consequence
            )
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (X, y) in val_loader:
                distance = X[:, 0:1]
                biotype = X[:, 1].long()
                ref_aa = X[:, 2].long()
                alt_aa = X[:, 3].long()
                consequence = X[:, 4:]
                y_hat = model(
                    distance,
                    biotype=biotype,
                    ref_aa=ref_aa,
                    alt_aa=alt_aa,
                    consequence=consequence
                )
                loss = loss_fn(y_hat, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            print("Best val loss improved, saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "output/baseline_model_best.pth")

if __name__ == "__main__":
    main()
