"""
Script to save all VEP data to pickle file in batches of 100.
"""

import sys
import time
from pathlib import Path
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, Future
import pandas as pd
from requests import Timeout
from urllib3.exceptions import ConnectTimeoutError
from utils import get_vep_data

def save_batch(vep_batch: list[dict], save_path: Path) -> None:
    """
    Save a batch of VEP data to a pickle file.
    Args:
        vep_batch (list[dict]): List of VEP data dictionaries
        save_path (Path): Path to save the pickle file
    Returns:
        None
    """
    with open(save_path, "ab") as f:
        pickle.dump(vep_batch, f)
    logging.info("Saved successfully")

def process_batch(
        start_index: int,
        end_index: int,
        ensp: pd.Series,
        pos: pd.Series,
        alt_long: pd.Series
    ) -> list[dict]:
    """
    Process a batch of VEP data.
    Args:
        start_index (int): The index of the data in the original DataFrame
        end_index (int): The ending index of the data in the original DataFrame
        ensp (str): Ensembl protein ID
        pos (int): Position of the variant
        alt_long (str): Alternate amino acid
    Returns:
        list[dict]: A list of VEP data dictionaries
    """
    vep_batch: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures: list[Future] = []
        for i in range(start_index, end_index):
            if i % 4 == 3:
                time.sleep(2)
            futures.append(
                executor.submit(get_vep_data_threaded, i, ensp.iat[i], pos.iat[i], alt_long.iat[i])
            )
    sorted_results = sorted([future.result() for future in futures], key=lambda x: x[0])
    error_occured: bool = False
    for result in sorted_results:
        if result[0] < 0:
            logging.error("Failed to fetch VEP data for index %d", 1-result[0])
            error_occured = True
    if not error_occured:
        for result in sorted_results:
            vep_batch.append(result[1])
        return vep_batch
    raise ValueError("Error occured fetching VEP data. See log for details.")

def get_vep_data_threaded(index: int, ensp: str, pos: int, alt_long: str) -> tuple[int, dict]:
    """
    Fetch VEP data in a thread and return the result with its index.
    Args:
        index (int): The index of the data in the original DataFrame
        ensp (str): Ensembl protein ID
        pos (int): Position of the variant
        alt_long (str): Alternate amino acid
    Returns:
        tuple(int, dict): A tuple containing the index and the fetched VEP data dictionary
        index is returned as -1 if there was an error fetching the data
        index is returned as -2 if there was timeout or connection error
    """
    retries_left: int = 10
    error_msg: str = ""
    while retries_left > 0:
        try:
            vep_data: dict = get_vep_data(ensp, pos, alt_long)
            if retries_left < 10:
                logging.info("Successfully fetched VEP data for index %d " \
                "after retrying %d times", index, 10 - retries_left)
            return (index, vep_data)
        except (Timeout, ConnectionError, ConnectTimeoutError) as e:
            logging.warning("Connection error fetching VEP data for index %d: %s", index, e)
            retries_left -= 1
            if retries_left <= 0:
                break
            time.sleep(2)
            error_msg = str(e)
        except Exception as e:
            logging.error("Error fetching VEP data for index %d: %s", index, e)
            error_msg = str(e)
    return (-index-1, {"error": error_msg})

def main(*args) -> None:
    """
    Main function to fetch and save VEP data
    Args:
        *args: Command line arguments
            1. csv to read data from
            2. save file location
            3. start row index (inclusive, starts at 0)
    Returns:
        None
    """
    if len(args) != 3:
        print("Usage: python vep.py <input_csv> <output_pickle> <start_index>")
        sys.exit(1)

    logging.basicConfig(filename=Path("notebooks/logs/vep.log"), level=logging.INFO, filemode="a",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    csv_path: Path = Path(args[0])
    save_path: Path = Path(args[1])
    start_index: int = int(args[2])
    logging.info("CLAs: %s %s %d", csv_path, save_path, start_index)

    df: pd.DataFrame = pd.read_csv(
        csv_path,
        usecols=["accession", "scoreset", "ensp", "pos", "ref_long", "alt_long"],
        dtype={"accession": str, "scoreset": str, "ensp": str,
               "pos": int, "ref_long": str, "alt_long": str}
    )

    i: int = start_index
    last_index_save: int = start_index
    try:
        vep_batch: list[dict] = []
        while i < len(df):
            end_index = min(i + 100, len(df)) # this index is exclusive
            logging.info("Processing batch: %d-%d/%d", i, end_index - 1, len(df))
            vep_batch = process_batch(i, end_index, df["ensp"], df["pos"], df["alt_long"])
            i += 100
            logging.info("Saving index %d through %d.", last_index_save, end_index - 1)
            last_index_save = i
            save_batch(vep_batch, save_path)
            vep_batch.clear()
            print()
        logging.info("All done! Processed up through index %d.", last_index_save)
        sys.exit(0)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        logging.warning("Process interrupted by user at index %d. " \
            "Last saved index was %d. Start from index %d to continue work.",
            i, last_index_save, last_index_save + 1
        )
        sys.exit(0)
    except (Timeout, ConnectionError) as e:
        logging.warning("Connection error at index %d: %s. Trying again.", i, e)
        main(csv_path, save_path, str(i))
    except ValueError as e:
        logging.error("Error processing row %d: %s", i, e)
    except Exception as e:
        logging.error("Unexpected error: %s", e)
    logging.info("Processed up through index %d before error. " \
        "Start from index %d to continue work.",
        last_index_save, last_index_save + 1
    )
    sys.exit(1)

if __name__ == "__main__":
    main(*sys.argv[1:])
