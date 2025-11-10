import torch
import torch.nn as nn

class DumbModel(nn.Module):
    """
    A single-layer baseline for data pipeline sanity checks.
    Takes only float input features (ignores categorical), outputs a scalar.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, X, **kwargs):
        # Ignore all categorical/multi-hot features for now
        # X: [batch, input_dim]
        return self.linear(X).squeeze(-1)


'''
Usage Example:
from models.dumb_model import DumbModel

# Use only the float/fused part of your data (e.g., embedding distance, etc.)
input_dim = 1 if fuse_mode == "distance" else dataset.X.shape[1]
model = DumbModel(input_dim=input_dim)

# In training loop:
y_hat = model(X)     # Only passes X, ignores kwargs like biotype, ref_aa, etc.

note: this is a minimal model for sanity checking the data pipeline.
jawn is stupid so don't use it for real training lmao
'''