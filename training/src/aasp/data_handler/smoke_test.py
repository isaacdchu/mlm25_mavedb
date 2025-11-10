import sys, os

# Go up 4 directories from this file → project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

print("✅ Added to sys.path:", ROOT_DIR)

from training.src.aasp.data_handler.aasp_dataset import AASPDataset
from torch.utils.data import DataLoader


# If you don't have the real pickle yet, create a tiny dummy one:
# (list of 3 records with 8-dim embeddings)
import pickle, numpy as np, os
os.makedirs("data/train", exist_ok=True)
dummy = []
for i in range(64):
    dummy.append({
        "accession": f"A{i}",
        "scoreset": "urn:mavedb:00000001-a-1",
        "ensp": "ENSP000000",
        "pos": i,
        "ref_long": "Ala",
        "alt_long": "Val",
        "score": float(np.random.randn()),
        "ref_embedding": np.random.randn(8).astype("float32"),
        "alt_embedding": np.random.randn(8).astype("float32"),
        "biotype": "protein_coding",
        "consequence": ["missense_variant"]
    })
with open("data/train/combined_train_data.pkl", "wb") as f:
    pickle.dump(dummy, f)

ds = AASPDataset(config_path="training/src/aasp/data_handler/config.yaml", fuse_mode="concat")
print("len:", len(ds), "sample shapes:", ds[0][0].shape, ds[0][1].shape)

loader = DataLoader(ds, batch_size=16, shuffle=True)
xb, yb = next(iter(loader))
print("batch:", xb.shape, yb.shape)
