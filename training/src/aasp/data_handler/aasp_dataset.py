from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.aasp.data_handler.data_handler import AASPConfig, AASPDataHandler

class AASPDataset(Dataset):
    """
    Flexible PyTorch Dataset for AASP baseline/feature eng models.
    Features selected/configured via config.yaml and __init__ args:
      - ref_embedding, alt_embedding (vectors)
      - embedding similarity/distance with configurable metric
      - categorical fields (biotype, consequence, or more)
      - score (float) is always included as target
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        fields: Optional[Sequence[str]] = None,      # ["ref_embedding", "alt_embedding", "biotype", ...]
        fuse_mode: str = "concat",                   # "concat", "subtract", "distance"
        embed_metric: str = "cosine",                # "cosine", "euclidean", "manhattan"
        categorical_config: Optional[Dict] = None,   # e.g., {"biotype": "embedding", "consequence": "multi_hot"}
        pad_to: Optional[int] = None,
        truncate_to: Optional[int] = None
    ):
        super().__init__()
        self.cfg = AASPConfig(config_path)
        self.handler = AASPDataHandler(self.cfg)

        # Load raw data
        self.records = self.handler.load_pickle(self.cfg.file_path)

        # Select fields as needed (inputs only, score always kept)
        self.records = self.handler.select_fields(self.records, fields)

        # ---- Embedding Features ----
        ref = self.handler.get_embedding(self.records, "ref_embedding", pad_to=pad_to, truncate_to=truncate_to)
        alt = self.handler.get_embedding(self.records, "alt_embedding", pad_to=pad_to, truncate_to=truncate_to)

        # Configurable fusion for embeddings: "concat" | "subtract" | "distance"
        if fuse_mode == "distance":
            X_embed = self.handler.compute_distance(self.records, method=embed_metric)        # [N, 1]
        elif fuse_mode == "subtract":
            X_embed = ref - alt                                                              # [N, D]
        elif fuse_mode == "concat":
            X_embed = np.concatenate([ref, alt], axis=1)                                     # [N, 2D]
        else:
            raise ValueError(f"Unknown fuse_mode: {fuse_mode}")

        X_feats = [X_embed]

        # ---- Categorical Features (modular config) ----
        # e.g., categorical_config={"biotype": "embedding", "consequence": "multi_hot"}
        self.vocabs = {}
        if categorical_config:
            for cat, encoding in categorical_config.items():
                if encoding == "embedding":
                    vocab = self.handler.fit_vocab(self.records, cat)
                    ids = self.handler.encode(self.records, cat, vocab)
                    X_feats.append(ids.reshape(-1, 1))
                    self.vocabs[cat] = vocab
                elif encoding == "one_hot":
                    vocab = self.handler.fit_vocab(self.records, cat)
                    ids = self.handler.encode(self.records, cat, vocab)
                    one_hot = self.handler.one_hot(ids, num_classes=len(vocab))
                    X_feats.append(one_hot)
                    self.vocabs[cat] = vocab
                elif encoding == "multi_hot":
                    vocab = self.handler.fit_vocab(self.records, cat)
                    mh = self.handler.encode_multihot(self.records, cat, vocab)
                    X_feats.append(mh)
                    self.vocabs[cat] = vocab

        # ---- Final Features and Target ----
        X = np.concatenate(X_feats, axis=1)
        y = self.handler.get_target(self.records, "score")[:, 0]     # flatten to 1D

        # Store for use
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Features/labels batch ready
        return self.X[idx], self.y[idx]
'''
# In main script aka use case:
fields = ["ref_embedding", "alt_embedding", "biotype", "consequence", "ref_long", "alt_long", "scoreset"]
cat_config = {
    "biotype": "embedding",
    "ref_long": "embedding",
    "alt_long": "embedding",
    "scoreset": "embedding",      
    "consequence": "multi_hot"
}
dataset = AASPDataset(
    config_path="config.yaml",
    fields=fields,
    fuse_mode="distance",
    embed_metric="cosine",
    categorical_config=cat_config
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

'''