"""
Helper script for getting all ESM C embeddings for a given dataaset
"""
import sys
import pandas as pd
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from torch import Tensor
import pickle
import utils

def save_batch(embeddings: dict[int, Tensor], save_path: Path) -> None:
    """
    Save a batch of embeddings to a pickle file.

    Args:
        embeddings (dict[int, Tensor]): A dictionary mapping indices to their corresponding embeddings.
        save_path (Path): The path where the embeddings should be saved.
    """
    with open(save_path, "ab") as f:
        pickle.dump(embeddings, f)

def main(*argv) -> None:
    """
    Main function to get ESM C embeddings for the training dataset and save them.
    """
    if (len(argv) != 3):
        print("Usage: python get_esm_c_embeddings.py <path/to/csv> <path/to/save/embeddings.pkl> <start_index>")
        sys.exit(1)
    logging.basicConfig(filename=Path("notebooks/logs/esm.log"), level=logging.INFO, filemode="a",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    csv_path: Path = Path(argv[0])
    save_path: Path = Path(argv[1])
    start_index: int = int(argv[2])
    logging.info("CLAs: %s %s %d", csv_path, save_path, start_index)

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["ensp", "pos", "ref_short", "alt_short"])
    

if __name__ == "__main__":
    main(*sys.argv[1:])
