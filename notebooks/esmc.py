"""
Helper script for getting all ESM C embeddings for a given dataaset
"""

import sys
import logging
from pathlib import Path
import pickle
import pandas as pd
from torch import Tensor
import utils

ensp_sequence_map: dict[str, str] = utils.pickle_to_dict(Path("data/ensp_sequence_map.pkl"))
ensp_embeddings_map: dict[str, Tensor] = utils.pickle_to_dict(Path("data/ensp_embeddings_map.pkl"))

def save_batch(embeddings: list[Tensor], save_path: Path) -> None:
    """
    Save a batch of embeddings to a pickle file.

    Args:
        embeddings (list[Tensor]): A list of tensors representing embeddings.
        save_path (Path): The path where the embeddings should be saved.
    """
    with open(save_path, "ab") as f:
        pickle.dump(embeddings, f)
    logging.info("Saved successfully")

def process_batch(
        start_index: int,
        end_index: int,
        ensp: pd.Series,
        pos: pd.Series,
        alt_short: pd.Series
    ) -> list[Tensor]:
    """
    Process a batch of data to get their ESM C embeddings.

    Args:
        start_index (int): The starting index of the batch (inclusive).
        end_index (int): The ending index of the batch (exclusive).
        ensp (pd.Series): Series containing Ensembl protein IDs.
        pos (pd.Series): Series containing positions of the variants.
        alt_short (pd.Series): Series containing alternate amino acids.

    Returns:
        list[Tensor]: A list of tensors representing the embeddings.
    """
    embeddings_batch: list[Tensor] = []
    for i in range(start_index, end_index):
        protein_id: str = ensp[i]
        position: int = pos[i]
        alt_aa: str = alt_short[i]
        ref_sequence: str = ensp_sequence_map.get(protein_id, "")
        if not ref_sequence:
            logging.error("Protein ID %s not found in sequence map at index %d.", protein_id, i)
            raise ValueError(f"Protein ID {protein_id} not found in sequence map.")
        alt_sequence: str = utils.sequence_substitution(ref_sequence, position, alt_aa)
        embeddings: Tensor = utils.get_embedding(alt_sequence)
        embeddings_batch.append(embeddings)
    return embeddings_batch

def main(*argv) -> None:
    """
    Main function to get ESM C embeddings for the training dataset and save them.
    Usage: python get_esm_c_embeddings.py <path/to/csv> <path/to/save/embeddings.pkl> <start_index>
    """
    if len(argv) != 3:
        print("Usage: python get_esm_c_embeddings.py <path/to/csv> "
            "<path/to/save/embeddings.pkl> <start_index>")
        sys.exit(1)
    logging.basicConfig(filename=Path("notebooks/logs/esm.log"), level=logging.INFO, filemode="a",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    csv_path: Path = Path(argv[0])
    save_path: Path = Path(argv[1])
    start_index: int = int(argv[2])
    logging.info("CLAs: %s %s %d", csv_path, save_path, start_index)
    logging.info("Using model: %s on device: %s", utils.MODEL_TYPE, utils.DEVICE)

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["ensp", "pos", "alt_short"])

    i: int = start_index
    last_index_save: int = start_index
    try:
        embeddings_batch: list[Tensor] = []
        while i < len(df):
            end_index: int = min(i + 100, len(df)) # this index is exclusive
            logging.info("Processing batch: %d-%d/%d", i, end_index - 1, len(df))
            embeddings_batch = process_batch(i, end_index, df["ensp"], df["pos"], df["alt_short"])
            i += 100
            logging.info("Saving index %d through %d.", last_index_save, end_index - 1)
            last_index_save = i - 1
            save_batch(embeddings_batch, save_path)
            embeddings_batch.clear()
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user at index %d. " \
            "Last saved index was %d. Start from index %d to continue work.",
            i, last_index_save, last_index_save + 1
        )
        sys.exit(0)
    except (ValueError, RuntimeError) as e:
        logging.error("Error processing index %d: %s", i, str(e))

if __name__ == "__main__":
    main(*sys.argv[1:])
