'''
Utility functions and constants for data processing and API interactions.
'''

import os
from pathlib import Path
import requests

# All file paths
RAW_TRAIN_PATH: os.PathLike = Path("../data/train/raw_train.csv")
RAW_TEST_PATH: os.PathLike = Path("../data/test/raw_test.csv")
SEQUENCE_MAP_PATH: os.PathLike = Path("../data/train/processed_train.csv")
PROCESSED_TEST_PATH: os.PathLike = Path("../data/test/processed_test.csv")

# API endpoints
MAVEDB_API = "https://api.mavedb.org/"
ENSEMBL_API = "https://rest.ensembl.org"

# Configuration constants
TIMEOUT = 3  # seconds

# Helper functions
def get_full_sequence(raw_ensp: str, known_map: dict[str, str]) -> str:
    """
    Fetch the full protein sequence from Ensembl given an Ensembl Protein ID.
    Args:
        ensp (str): Ensembl Protein ID (ENSP00000XXXXXX.X)
        known_map (dict[str, str]): A mapping of known Ensembl IDs to their sequences to avoid redundant API calls
            e.g., {"ENSP00000354587": "MEEPQSDPSV..."}
    Returns:
        str: Full protein sequence as a string
    """
    if raw_ensp in known_map:
        return known_map[raw_ensp]
    ensp = raw_ensp.split(".")[0]  # Remove version number if present
    response: requests.Response = requests.get(
        f"{ENSEMBL_API}/sequence/id/{ensp}",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT)
    if response.status_code != 200:
        raise ValueError(f"Error fetching sequence for {ensp}: {response.status_code}")
    sequence: str = response.json().get("seq", "")
    if not sequence:
        raise ValueError(f"No sequence found for {ensp}")
    known_map[raw_ensp] = sequence
    return sequence
