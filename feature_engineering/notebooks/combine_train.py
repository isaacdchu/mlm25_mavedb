"""
Script to combine various data sources into a single DataFrame for training.
Saves the combined DataFrame as a pickle file.
"""

from pathlib import Path
import pickle
import pandas as pd
from torch import Tensor

import utils

def main() -> None:
    """
    Combines data into a single DataFrame and saves it as a pickle file.
    Requires the following files:
    1. data/train/raw_train.csv (raw training data from kaggle)
    2. data/test/raw_test.csv (raw test data from kaggle)
    3. data/train/ensp_embeddings_map.pkl (list of ESMC embeddings for alternate sequences)
            (should be named esmc_embeddings.pkl but i messed up))
    4. data/train/train_vep.pkl (VEP data for training set)
    5. data/ensp_embeddings_map.pkl (maps ensp to ESMC embeddings for reference sequences)
    """
    print("Loading data...")
    train_df: pd.DataFrame = pd.read_csv(Path("../data/train/raw_train.csv"), usecols=[
        "accession", "scoreset", "ensp", "pos", "ref_long", "alt_long", "score"
    ], dtype={
        "accession": str, "scoreset": str, "ensp": str,
        "pos": int, "ref_long": str, "alt_long": str, "score": float
    })

    # add ref embedding column to train_df
    print("Adding reference embeddings...")
    ref_embeddings_map: dict[str, Tensor] = pickle.load(open(Path("../data/ensp_embeddings_map.pkl"), "rb"))
    # convert tensors into cpu device if they are not already
    for ensp in ref_embeddings_map:
        ref_embeddings_map[ensp] = ref_embeddings_map[ensp].cpu()
    train_df["ref_embedding"] = train_df["ensp"].map(ref_embeddings_map)

    # add alt embedding column to train_df
    print("Adding alternate embeddings...")
    train_embeddings: list[Tensor] = []
    with open(Path("../data/train/ensp_embeddings.pkl"), "rb") as f:
        while True:
            try:
                # convert tensors into cpu device if they are not already
                train_embeddings.extend(embedding.cpu() for embedding in pickle.load(f))
            except EOFError:
                break
    train_df["alt_embedding"] = train_embeddings

    # add vep data column to train_df
    print("Adding VEP data...")
    vep_data: list[dict] = utils.vep_from_pickle(Path("../data/train/train_vep.pkl"))
    biotype: list[str] = []
    consequences: list[list[str]] = []
    for data in vep_data:
        transcript_consequences: dict = data.get("transcript_consequences", [{}])[0]
        biotype.append(transcript_consequences.get("biotype", ""))
        consequences.append([consequence for consequence in transcript_consequences.get("consequence_terms", [])])
    train_df["biotype"] = biotype
    train_df["consequences"] = consequences

    # remove rows from train_df if its scoreset is not in test_df
    print("Removing irrelevant scoresets...")
    keep_scoresets: set[str] = set(pd.read_csv(Path("../data/test/raw_test.csv"))["scoreset"].unique())
    train_df = train_df[train_df["scoreset"].isin(keep_scoresets)].reset_index(drop=True)

    # save combined train_df
    print("Saving combined training data...")
    combined_save_path: Path = Path("../data/train/combined_train_data.pkl")
    with open(combined_save_path, "wb") as f:
        pickle.dump(train_df, f)
    print(f"Combined training data saved to {combined_save_path}")
    
if __name__ == "__main__":
    main()
