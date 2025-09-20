'''
Utility functions and constants for data processing and API interactions.
'''

import os
from pathlib import Path
import pickle
from typing import Any
import requests

# All file paths
RAW_TRAIN_PATH: os.PathLike = Path("../data/train/raw_train.csv")
RAW_TEST_PATH: os.PathLike = Path("../data/test/raw_test.csv")
PROCESSED_TEST_PATH: os.PathLike = Path("../data/test/processed_test.csv")
PROCESSED_TRAIN_PATH: os.PathLike = Path("../data/train/processed_train.csv")
TRAIN_ENSP_SEQUENCE_MAP_PATH: os.PathLike = Path("../data/train_ensp_sequence_map.pkl")
TEST_ENSP_SEQUENCE_MAP_PATH: os.PathLike = Path("../data/test_ensp_sequence_map.pkl")

# API endpoints
MAVEDB_API = "https://api.mavedb.org/"
ENSEMBL_API = "https://rest.ensembl.org"

# Configuration constants
TIMEOUT = 3  # seconds

# Helper functions
def get_full_sequence(raw_ensp: str, ensp_sequence_map: dict[str, str]) -> str:
    """
    Fetch the full protein sequence from Ensembl given an Ensembl Protein ID.
    Modifies ensp_sequence_map in place to cache results.
    Args:
        ensp (str): Ensembl Protein ID (ENSP00000XXXXXX.X)
        ensp_sequence_map (dict[str, str]): A mapping of known Ensembl IDs to their sequences
            to avoid redundant API calls. e.g., {"ENSP00000354587": "MEEPQSDPSV..."}
    Returns:
        str: Full protein sequence as a string
    Raises:
        ValueError: If the API request fails or returns an error status code
    """
    if raw_ensp in ensp_sequence_map:
        return ensp_sequence_map[raw_ensp]
    ensp: str = raw_ensp.split(".")[0]  # Remove version number if present
    response: requests.Response = requests.get(
        f"{ENSEMBL_API}/sequence/id/{ensp}",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT)
    if response.status_code != 200:
        raise ValueError(f"Error fetching sequence for {ensp}: {response.status_code}")
    sequence: str = response.json().get("seq", "")
    if not sequence:
        raise ValueError(f"No sequence found for {ensp}")
    ensp_sequence_map[raw_ensp] = sequence
    return sequence

def to_hgvs(raw_ensp: str, ref_long: str, pos: int, alt_long: str) -> str:
    """
    Convert variant information to HGVS notation.
    Args:
        raw_ensp (str): Ensembl Protein ID (ENSP00000XXXXXX.X)
        ref_long (str): Reference amino acid (eg Pro for P/Proline)
        pos (int): Position of the variant
        alt_long (str): Alternate amino acid (eg Leu for L/Leucine)
    Returns:
        str: HGVS notation (e.g., "ENSP00000354587.3:p.Pro175Leu")
    """
    return f"{raw_ensp}:p.{ref_long}{pos}{alt_long}"

def get_vep_data(hgvs: str) -> dict:
    """
    Fetch variant effect prediction data from Ensembl VEP API.
    Assumes that hgvs is correctly formatted.
    Args:
        hgvs (str): HGVS notation of the variant (e.g., "ENSP00000354587.3:p.R175H")
    Returns:
        dict: Parsed JSON response from the VEP API
    Raises:
        ValueError: If the API request fails or returns an error status code
    """
    response: requests.Response = requests.get(
        f"{ENSEMBL_API}/vep/human/hgvs/{hgvs}",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT)
    if response.status_code != 200:
        raise ValueError(f"Error fetching VEP data for {hgvs}: {response.status_code}")
    return response.json()

def dict_to_pickle(file_location: os.PathLike, dictionary: dict[Any, Any]) -> None:
    """
    Turns a dictionary into a pickle
    Args:
        file_location (str): file location of where you want to dump the dictionary
        dictionary (dict): dictionary that you want to pickle
    Returns:
        None
    """
    with open(file_location, "wb") as f:
        pickle.dump(dictionary, f)

def pickle_to_dict(file_location: os.PathLike) -> dict[Any, Any]:
    """
    Unpickles a pkl file into a dictionary
    Args:
        file_location (str): file location of where the pickle is 
    Returns:
        dict: dictionary of pickle
    """
    with open(file_location, "rb") as f:
        dictionary = pickle.load(f)
        return dictionary
