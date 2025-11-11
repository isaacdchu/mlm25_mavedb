# from src.aasp.model.models.dumb_model import DumbModel

# import torch
# from torch.utils.data import DataLoader

# from src.aasp.data_handler.aasp_dataset import AASPDataset


# def main():

#     # --- build dataset ---
#     dataset = AASPDataset(
#         config_path="config.yaml",
#         fields=["ref_embedding", "alt_embedding"],   # minimal
#         fuse_mode="distance",                        # gives X shape [N,1]
#         embed_metric="cosine"
#     )

#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # --- build model ---
#     input_dim = dataset.X.shape[1]        # >= 1
#     model = DumbModel(input_dim=input_dim)

#     # --- one pass sanity check ---
#     X, y = next(iter(loader))             # one batch
#     y_hat = model(X)                      # forward

#     loss = torch.nn.functional.mse_loss(y_hat, y)

#     print("batch y_hat shape:", y_hat.shape)
#     print("batch loss:", float(loss))


# if __name__ == "__main__":
#     main()

from src.aasp.model.models.dumb_model import DumbModel

import torch
from torch.utils.data import DataLoader

from src.aasp.data_handler.aasp_dataset import AASPDataset


def main():

    print(">>> building dataset...")
    dataset = AASPDataset(
        config_path="config.yaml",
        fields=["ref_embedding", "alt_embedding"],
        fuse_mode="distance",
        embed_metric="cosine"
    )

    print(">>> dataset built.")
    print(f"Dataset length: {len(dataset)}")
    print(f"Feature tensor shape: {dataset.X.shape}")   # [N, D]
    print(f"Target tensor shape:  {dataset.y.shape}")   # [N]

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(">>> dataloader constructed.")

    # --- build model ---
    input_dim = dataset.X.shape[1]
    model = DumbModel(input_dim=input_dim)
    print(f"Model created with input_dim = {input_dim}")

    # --- one batch sanity check ---
    X, y = next(iter(loader))
    print("\n--- sample batch ---")
    print(f"X batch shape: {X.shape}")
    print(f"y batch shape: {y.shape}")
    print(f"first 5 y values: {y[:5].tolist()}")

    # forward pass
    y_hat = model(X)
    print(f"y_hat shape: {y_hat.shape}")
    print(f"first 5 y_hat values: {y_hat[:5].detach().tolist()}")

    # quick loss
    loss = torch.nn.functional.mse_loss(y_hat, y)
    print(f"sample loss: {float(loss):.6f}")

    print("\n>>> dumb model sanity run complete — no crashes")


if __name__ == "__main__":
    main()
