"""
Helper script for getting all ESM C embeddings for a given dataaset
"""
import sys
import logging
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, Future
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

def get_embedding_threaded(index: int, ensp: str, pos: int, alt_short: str) -> tuple[int, Tensor]:
    """
    Calculate ESM C embedding for a given protein variant in a thread and return the result with its index.

    Args:
        index (int): The index of the data in the original DataFrame.
        ensp (str): Ensembl protein ID.
        pos (int): Position of the variant.
        alt_short (str): Alternate amino acid.

    Returns:
        tuple[int, Tensor]: A tuple containing the index and its corresponding embedding.
                            If an error occurs, returns a tuple with a negative index and an empty tensor.
    """
    retries_left: int = 5
    while (retries_left > 0):
        try:
            if ensp not in ensp_sequence_map:
                logging.error("Ensp %s not found in sequence map", ensp)
                return (-1 * (index + 1), Tensor())
            sequence: str = utils.sequence_substitution(ensp_sequence_map[ensp], pos, alt_short)
            embedding: Tensor = utils.get_embedding(sequence)
            if (embedding.numel() == 0):
                logging.error("Empty embedding for index %d", index)
                retries_left -= 1
                continue
            if (retries_left < 5):
                logging.info("Successfully fetched embedding for index %d after %d retries", index, 5-retries_left)
            return (index, embedding)
        except Exception as e:
            logging.error("Error fetching embedding for index %d: %s", index, str(e))
            retries_left -= 1
    # If we exhaust retries, return an error indicator
    logging.error("Exhausted retries for index %d", index)
    return (-1 * (index + 1), Tensor())

def process_batch(start_index: int, end_index: int, ensp: pd.Series, pos: pd.Series, alt_short: pd.Series) -> list[Tensor]:
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
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures: list[Future] = []
        for i in range(start_index, end_index):
            futures.append(executor.submit(get_embedding_threaded, i, ensp.iat[i], pos.iat[i], alt_short.iat[i]))
    sorted_results = sorted([future.result() for future in futures], key=lambda x: x[0])
    error_occured: bool = False
    for result in sorted_results:
        if (result[0] < 0):
            logging.error("Failed to fetch embedding for index %d", 1-result[0])
            error_occured = True
    if not error_occured:
        embeddings_batch = [result[1] for result in sorted_results]
        return embeddings_batch
    raise ValueError("Error occured fetching embeddings. See log for details.")

def main(*argv) -> None:
    """
    Main function to get ESM C embeddings for the training dataset and save them.
    Usage: python get_esm_c_embeddings.py <path/to/csv> <path/to/save/embeddings.pkl> <start_index>
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
        logging.warning("Process interrupted by user at index %d. Last saved index was %d. Start from index %d to continue work.", i, last_index_save, last_index_save + 1)
        sys.exit(0)
    except Exception as e:
        logging.error("Error processing index %d: %s", i, str(e))

if __name__ == "__main__":
    main(*sys.argv[1:])
