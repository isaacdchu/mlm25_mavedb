print("importing modules...")
import pandas as pd
import pickle
from torch import Tensor
import utils

# Read raw data
print("reading data..." )
raw_train: pd.DataFrame = pd.read_csv("../data/train/raw_train.csv")
raw_test: pd.DataFrame = pd.read_csv("../data/test/raw_test.csv", usecols=["scoreset"])
raw_train_vep: list[dict] = []
with open("../data/train/train_vep.pkl", "rb") as f:
    while True:
        try:
            raw_train_vep.append(pickle.load(f))
        except EOFError:
            break
raw_train_embeddings: list[Tensor] = []
with open("../data/train/ensp_embeddings.pkl", "rb") as f:
    while True:
        try:
            raw_train_embeddings.extend(pickle.load(f))
        except EOFError:
            break
# Filter out scoresets that are not in the test set
print("filtering data...")
# Create a map from ensp + pos + ref_long + alt_long to vep data for each row in training data
vep_map: dict = {}
index: int = 0
for _, row in raw_train.iterrows():
    KEY: str = utils.make_vep_key(
        ensp=row["ensp"],
        pos=row["pos"],
        ref_long=row["ref_long"],
        alt_long=row["alt_long"],
    )
    vep_map[KEY] = raw_train_vep[index]
    index += 1
# Create a map from ensp + pos + ref_long + alt_long to embedding data for each row in training data
embedding_map: dict = {}
index = 0
for _, row in raw_train.iterrows():
    KEY: str = utils.make_vep_key(
        ensp=row["ensp"],
        pos=row["pos"],
        ref_long=row["ref_long"],
        alt_long=row["alt_long"],
    )
    embedding_map[KEY] = raw_train_embeddings[index]
    index += 1
# Identify rows to keep
rows_to_keep: list[int] = []
test_scoresets: set = set(raw_test["scoreset"].unique())
i: int = 0
for idx, row in raw_train.iterrows():
    scoreset: str = row["scoreset"]
    if scoreset in test_scoresets:
        continue
    # Check if VEP data exists for this row
    vep_key: str = utils.make_vep_key(
        ensp=row["ensp"],
        pos=row["pos"],
        ref_long=row["ref_long"],
        alt_long=row["alt_long"],
    )
    if vep_key in vep_map:
        rows_to_keep.append(i)
    i += 1
# Create filtered training DataFrame
filtered_train: pd.DataFrame = raw_train.loc[rows_to_keep].reset_index(drop=True)
# Create filtered training VEP data mapping
filtered_train_vep: dict[str, dict] = {}
for idx, row in filtered_train.iterrows():
    vep_key: str = utils.make_vep_key(
        ensp=row["ensp"],
        pos=row["pos"],
        ref_long=row["ref_long"],
        alt_long=row["alt_long"],
    )
    filtered_train_vep[vep_key] = vep_map[vep_key]

# Create filtered training embeddings mapping
filtered_train_embeddings: dict[str, Tensor] = {}
for idx, row in filtered_train.iterrows():
    embedding_key: str = utils.make_vep_key(
        ensp=row["ensp"],
        pos=row["pos"],
        ref_long=row["ref_long"],
        alt_long=row["alt_long"],
    )
    filtered_train_embeddings[embedding_key] = embedding_map[embedding_key]

# Save filtered training data
print("saving processed data...")
SEED: int = 0
randomized_train = filtered_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
randomized_train.to_csv("../data/train/shuffled_processed_train.csv", index=False)
filtered_train.to_csv("../data/train/processed_train.csv", index=False)
with open("../data/train/processed_train_vep.pkl", "wb") as f:
    pickle.dump(filtered_train_vep, f)
with open("../data/train/processed_train_embeddings.pkl", "wb") as f:
    pickle.dump(filtered_train_embeddings, f)
print("done.")
