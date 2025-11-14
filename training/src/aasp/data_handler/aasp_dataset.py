# from __future__ import annotations
# from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from .data_handler import AASPConfig, AASPDataHandler

# class AASPDataset(Dataset):
#     """
#     Flexible PyTorch Dataset for AASP baseline/feature eng models.
#     Features selected/configured via config.yaml and __init__ args:
#       - ref_embedding, alt_embedding (vectors)
#       - embedding similarity/distance with configurable metric
#       - categorical fields (biotype, consequence, or more)
#       - score (float) is always included as target
#     """

#     def __init__(
#         self,
#         config_path: str = "config.yaml",
#         fields: Optional[Sequence[str]] = None,      # ["ref_embedding", "alt_embedding", "biotype", ...]
#         fuse_mode: str = "concat",                   # "concat", "subtract", "distance"
#         embed_metric: str = "cosine",                # "cosine", "euclidean", "manhattan"
#         categorical_config: Optional[Dict] = None,   # e.g., {"biotype": "embedding", "consequence": "multi_hot"}
#         pad_to: Optional[int] = None,
#         truncate_to: Optional[int] = None
#     ):
#         super().__init__()
#         self.cfg = AASPConfig(config_path)
#         self.handler = AASPDataHandler(self.cfg)

#         # Load raw data
#         self.records = self.handler.load_pickle(self.cfg.file_path)

#         # Select fields as needed (inputs only, score always kept)
#         self.records = self.handler.select_fields(self.records, fields)

#         # ---- Embedding Features ----
#         ref = self.handler.get_embedding(self.records, "ref_embedding", pad_to=pad_to, truncate_to=truncate_to)
#         alt = self.handler.get_embedding(self.records, "alt_embedding", pad_to=pad_to, truncate_to=truncate_to)

#         # Configurable fusion for embeddings: "concat" | "subtract" | "distance"
#         if fuse_mode == "distance":
#             X_embed = self.handler.compute_distance(self.records, method=embed_metric)        # [N, 1]
#         elif fuse_mode == "subtract":
#             X_embed = ref - alt                                                              # [N, D]
#         elif fuse_mode == "concat":
#             X_embed = np.concatenate([ref, alt], axis=1)                                     # [N, 2D]
#         else:
#             raise ValueError(f"Unknown fuse_mode: {fuse_mode}")

#         X_feats = [X_embed]

#         # ---- Categorical Features (modular config) ----
#         # e.g., categorical_config={"biotype": "embedding", "consequence": "multi_hot"}
#         self.vocabs = {}
#         if categorical_config:
#             for cat, encoding in categorical_config.items():
#                 if encoding == "embedding":
#                     vocab = self.handler.fit_vocab(self.records, cat)
#                     ids = self.handler.encode(self.records, cat, vocab)
#                     X_feats.append(ids.reshape(-1, 1))
#                     self.vocabs[cat] = vocab
#                 elif encoding == "one_hot":
#                     vocab = self.handler.fit_vocab(self.records, cat)
#                     ids = self.handler.encode(self.records, cat, vocab)
#                     one_hot = self.handler.one_hot(ids, num_classes=len(vocab))
#                     X_feats.append(one_hot)
#                     self.vocabs[cat] = vocab
#                 elif encoding == "multi_hot":
#                     vocab = self.handler.fit_vocab(self.records, cat)
#                     mh = self.handler.encode_multihot(self.records, cat, vocab)
#                     X_feats.append(mh)
#                     self.vocabs[cat] = vocab

#         # ---- Final Features and Target ----
#         X = np.concatenate(X_feats, axis=1)
#         y = self.handler.get_target(self.records, "score")[:, 0]     # flatten to 1D

#         # Store for use
#         self.X = torch.from_numpy(X).float()
#         self.y = torch.from_numpy(y).float()

#     def __len__(self) -> int:
#         return self.X.shape[0]

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Features/labels batch ready
#         return self.X[idx], self.y[idx]
# '''
# # In main script aka use case:
# fields = ["ref_embedding", "alt_embedding", "biotype", "consequence", "ref_long", "alt_long", "scoreset"]
# cat_config = {
#     "biotype": "embedding",
#     "ref_long": "embedding",
#     "alt_long": "embedding",
#     "scoreset": "embedding",      
#     "consequence": "multi_hot"
# }
# dataset = AASPDataset(
#     config_path="config.yaml",
#     fields=fields,
#     fuse_mode="distance",
#     embed_metric="cosine",
#     categorical_config=cat_config
# )
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# '''

# training/src/aasp/data_handler/aasp_dataset.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_handler import AASPConfig, AASPDataHandler


class AASPDataset(Dataset):
    """
    PyTorch Dataset that turns your pickle records into tensors for models.

    What it can build:
      • Embedding features from ref/alt embeddings with a chosen fusion:
          - "concat":  [ref, alt]          -> [N, 2D]
          - "subtract":[ref - alt]          -> [N, D]
          - "distance":handler.compute_distance(...) -> [N, 1]
      • Optional categorical features (per config):
          - "embedding": integer ids (you will feed to nn.Embedding in your model)
          - "one_hot"  : one-hot vectors
          - "multi_hot": multi-hot vectors for list-valued fields (e.g., consequence)
      • Target y (score) as float tensor

    Notes
    -----
    - We use AASPDataHandler for all parsing/encoding logic so this stays thin.
    - If your embeddings vary in length, pass `pad_to` or `truncate_to`.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        fields: Optional[Sequence[str]] = None,      # e.g. ["ref_embedding","alt_embedding","biotype","consequence"]
        fuse_mode: str = "concat",                   # "concat" | "subtract" | "distance"
        embed_metric: str = "cosine",                # used only if fuse_mode == "distance"
        categorical_config: Optional[Dict[str, str]] = None,
        pad_to: Optional[int] = None,
        truncate_to: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = AASPConfig(config_path)
        self.handler = AASPDataHandler(self.cfg)

        # ---- Load & shape records
        records = self.handler.load_pickle(self.cfg.file_path)
        records = self.handler.select_fields(records, fields)

        # ---- Embedding features
        # These keys are assumed to exist in records if you request them.
        if fuse_mode == "distance":
            X_embed = self.handler.compute_distance(
                records,
                method=embed_metric,
                ref_key="ref_embedding",
                alt_key="alt_embedding",
                pad_to=pad_to,
                truncate_to=truncate_to,
            )  # [N, 1]
        else:
            ref = self.handler.get_embedding(records, "ref_embedding", pad_to=pad_to, truncate_to=truncate_to)
            alt = self.handler.get_embedding(records, "alt_embedding", pad_to=pad_to, truncate_to=truncate_to)
            if fuse_mode == "subtract":
                X_embed = ref - alt                         # [N, D]
            elif fuse_mode == "concat":
                X_embed = np.concatenate([ref, alt], 1)     # [N, 2D]
            else:
                raise ValueError(f"Unknown fuse_mode: {fuse_mode!r}")

        feature_blocks: List[np.ndarray] = [X_embed]
        self.vocabs: Dict[str, Mapping[str, int]] = {}

        # ---- Optional categorical features
        # e.g., {"biotype": "embedding", "scoreset": "embedding", "consequence": "multi_hot"}
        if categorical_config:
            for name, mode in categorical_config.items():
                if mode == "embedding":
                    vocab = self.handler.fit_vocab(records, name)
                    ids = self.handler.encode(records, name, vocab)         # [N]
                    feature_blocks.append(ids.reshape(-1, 1))               # store ids as a feature column
                    self.vocabs[name] = vocab
                elif mode == "one_hot":
                    vocab = self.handler.fit_vocab(records, name)
                    ids = self.handler.encode(records, name, vocab)         # [N]
                    oh = self.handler.one_hot(ids, num_classes=len(vocab))  # [N, C]
                    feature_blocks.append(oh)
                    self.vocabs[name] = vocab
                elif mode == "multi_hot":
                    vocab = self.handler.fit_vocab(records, name)
                    mh = self.handler.encode_multihot(records, name, vocab) # [N, C]
                    feature_blocks.append(mh)
                    self.vocabs[name] = vocab
                else:
                    raise ValueError(f"Unknown categorical mode for {name!r}: {mode!r}")

        # ---- Final tensors
        X = np.concatenate(feature_blocks, axis=1)
        y = self.handler.get_target(records, "score")[:, 0]

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

        # (optional) keep some metadata
        self.num_samples = self.X.shape[0]
        self.input_dim = self.X.shape[1]
        self.fuse_mode = fuse_mode
        self.embed_metric = embed_metric

    # ---- PyTorch Dataset API
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
