"""
Script to save all VEP data to pickle file in batches of 100.
"""

import sys
from pathlib import Path
import logging
import pickle
import pandas as pd
from requests import Timeout
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
    if (len(args) != 3):
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
        dtype={"accession": str, "scoreset": str, "ensp": str, "pos": int, "ref_long": str, "alt_long": str}
    )

    i: int = start_index
    last_index_save: int = 0
    try:
        vep_batch: list[dict] = []
        while (i < len(df)):
            print(f"{i}/{len(df)} ", end="")
            vep_batch.append(
                get_vep_data(
                    df["ensp"].iat[i],
                    df["pos"].iat[i],
                    df["alt_long"].iat[i]
                )
            )
            i += 1
            if (i % 100 == 0):
                logging.info("Saving index %d through %d.", last_index_save, i - 1)
                last_index_save = i - 1
                save_batch(vep_batch, save_path)
                vep_batch.clear()
            print()
        logging.info("Saving index %d through %d.", last_index_save, i - 1)
        last_index_save = i - 1
        save_batch(vep_batch, save_path)
        vep_batch.clear()
        logging.info("All done! Processed up through index %d.", last_index_save)
        sys.exit(0)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        logging.warning("Process interrupted by user at index %d. Last saved index was %d. Start from index %d to continue work.", i, last_index_save, last_index_save + 1)
        sys.exit(0)
    except (Timeout, ConnectionError) as e:
        logging.warning("Connection error at index %d: %s. Trying again.", i, e)
        main(csv_path, save_path, str(i))
    except ValueError as e:
        logging.error("Error processing row %d: %s", i, e)
    except Exception as e:
        logging.error("Unexpected error: %s", e)
    logging.info("Processed up through index %d before error. Start from index %d to continue work.", last_index_save, last_index_save + 1)
    sys.exit(1)

if __name__ == "__main__":
    main(*sys.argv[1:])