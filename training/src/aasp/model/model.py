import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cat_dims: dict = None,                 # e.g. {"biotype": (num_categories, embedding_dim)}
        multi_hot_dims: dict = None,           # e.g. {"consequence": num_categories}
        hidden_dims: tuple = (64, 16),
        dropout_rates: tuple = (0.2, 0.1)
    ):
        super().__init__()
        self.cat_embeddings = nn.ModuleDict()
        self.multi_hot_layers = nn.ModuleDict()

        # Standard categorical embeddings (biotype, ref_aa, alt_aa, scoreset)
        for name, (num_cats, embed_dim) in cat_dims.items():
            self.cat_embeddings[name] = nn.Embedding(num_cats, embed_dim)

        for name, out_dim in multi_hot_dims.items():
            self.multi_hot_layers[name] = nn.Linear(out_dim, out_dim)

        concat_dim = input_dim + sum(e[1] for e in cat_dims.values()) + sum(multi_hot_dims.values())

        # Simple feedforward MLP head
        self.network = nn.Sequential(
            nn.Linear(concat_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, X, **features):
        # X: embedding fusion/distance and any raw float features [batch, input_dim]
        batch = X
        # Categorical (embedding) features
        for name, emb in self.cat_embeddings.items():
            # features[name]: [batch] (ints)
            batch = torch.cat([batch, emb(features[name])], dim=1)
        # Multi-hot features
        for name, layer in self.multi_hot_layers.items():
            # features[name]: [batch, n]
            batch = torch.cat([batch, layer(features[name])], dim=1)
        out = self.network(batch)
        return out.squeeze(-1)

# Usage Example:
# input_dim = 1 (distance) or 2D/embedding concat size
# cat_dims = {"biotype": (num_classes_biotype, 4), "ref_aa": (num_classes_aa, 4), "alt_aa": (num_classes_aa, 4)}
# multi_hot_dims = {"consequence": num_consequence}
# model = BaselineModel(input_dim, cat_dims, multi_hot_dims)
'''
model = BaselineModel(
    input_dim=X_dim,                # e.g., embedding fusion (1 for distance, 1280 for ESM embed, 2560 for concat)
    cat_dims={
        "biotype": (len(biotype_vocab), 4),
        "ref_aa": (len(aa_vocab), 4),
        "alt_aa": (len(aa_vocab), 4),
        "scoreset": (len(scoreset_vocab), 4)
    },
    multi_hot_dims={"consequence": len(consequence_vocab)},
    hidden_dims=(64, 16),
    dropout_rates=(0.2, 0.1)
)


# Forward pass
# Supply X and named categorical/multi-hot features output from AASPDataset:
y_hat = model(
    X,                                # [batch, X_dim]
    biotype=biotype_ids,              # [batch]
    ref_aa=ref_aa_ids,                # [batch]
    alt_aa=alt_aa_ids,                # [batch]
    scoreset=scoreset_ids,            # [batch]
    consequence=consequence_multihot  # [batch, num_consequence]
)
'''
