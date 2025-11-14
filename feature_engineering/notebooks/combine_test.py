"""
Script to combine various data sources into a single DataFrame for testing.
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
    1. data/test/raw_test.csv (raw test data from kaggle)
    2. data/test/ensp_embeddings_map.pkl (list of ESMC embeddings for alternate sequences)
            (should be named esmc_embeddings.pkl but i messed up))
    3. data/test/test_vep.pkl (VEP data for test set)
    4. data/ensp_embeddings_map.pkl (maps ensp to ESMC embeddings for reference sequences)
    """
    print("Loading data...")
    test_df: pd.DataFrame = pd.read_csv(Path("../data/test/raw_test.csv"), usecols=[
        "accession", "scoreset", "ensp", "pos", "ref_long", "alt_long", 
    ], dtype={
        "accession": str, "scoreset": str, "ensp": str,
        "pos": int, "ref_long": str, "alt_long": str
    })

    # add ref embedding column to test_df
    print("Adding reference embeddings...")
    ref_embeddings_map: dict[str, Tensor] = pickle.load(open(Path("../data/ensp_embeddings_map.pkl"), "rb"))
    # convert tensors into cpu device if they are not already
    for ensp in ref_embeddings_map:
        ref_embeddings_map[ensp] = ref_embeddings_map[ensp].cpu()
    test_df["ref_embedding"] = test_df["ensp"].map(ref_embeddings_map)

    # add alt embedding column to test_df
    print("Adding alternate embeddings...")
    test_embeddings: list[Tensor] = []
    with open(Path("../data/test/ensp_embeddings.pkl"), "rb") as f:
        while True:
            try:
                # convert tensors into cpu device if they are not already
                test_embeddings.extend(embedding.cpu() for embedding in pickle.load(f))
            except EOFError:
                break
    test_df["alt_embedding"] = test_embeddings

    # add vep data column to test_df
    print("Adding VEP data...")
    vep_data: list[dict] = utils.vep_from_pickle(Path("../data/test/test_vep.pkl"))
    biotype: list[str] = []
    consequences: list[list[str]] = []
    for data in vep_data:
        transcript_consequences: dict = data.get("transcript_consequences", [{}])[0]
        biotype.append(transcript_consequences.get("biotype", ""))
        consequences.append([consequence for consequence in transcript_consequences.get("consequence_terms", [])])
    test_df["biotype"] = biotype
    test_df["consequences"] = consequences
    
    # save combined test_df
    print("Saving combined test data...")
    combined_save_path: Path = Path("../data/test/combined_test_data.pkl")
    with open(combined_save_path, "wb") as f:
        pickle.dump(test_df, f)
    print(f"Combined test data saved to {combined_save_path}")

if __name__ == "__main__":
    main()
