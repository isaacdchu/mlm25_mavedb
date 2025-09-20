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
TRAIN_ENSP_SEQUENCE_MAP_PATH: os.PathLike = Path("../data/train/train_ensp_sequence_map.pkl")
TEST_ENSP_SEQUENCE_MAP_PATH: os.PathLike = Path("../data/test/test_ensp_sequence_map.pkl")

# API endpoints
MAVEDB_API = "https://api.mavedb.org/"
ENSEMBL_API = "https://rest.ensembl.org"

# Configuration constants
TIMEOUT = 10  # seconds

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

def get_nucleotide_change(ref_nucleotides: str, alt_long: str, strand: int) -> str:
    """
    Parses strand and alt_long to return a nucleotide sequence.
    Args:
        strand (int): Strand information (1 or -1)
        alt_long (str): Alternate amino acid (eg Leu for L/Leucine)
    Returns:
        str: Nucleotide sequence of alt_long
    """
    # Note: there are multiple codons for each amino acid; we choose one arbitrarily
    # Nucleotide-level differences are insignificant for our purposes
    # TODO: handle cases where ref_nucleotides == alt_nucleotides
    # Strand is 1
    protein_to_codon_p1: dict[str, str] = {
        "Arg": "CGA", "Ala": "GCA", "Asn": "AAC", "Asp": "GAC",
        "Cys": "TGC", "Gln": "CAA", "Glu": "GAA", "Gly": "GGA",
        "His": "CAC", "Ile": "ATA", "Leu": "CTA", "Lys": "AAA",
        "Met": "ATG", "Phe": "TTC", "Pro": "CCA", "Ser": "TCA",
        "Thr": "ACA", "Trp": "TGG", "Tyr": "TAC", "Val": "GTA",
        "Ter": "TAA"  # Stop codon
    }
    # Strand is -1
    protein_to_codon_n1: dict[str, str] = {
        "Arg": "ACG", "Ala": "TGC", "Asn": "GTT", "Asp": "GTC",
        "Cys": "GCA", "Gln": "TTG", "Glu": "TTC", "Gly": "TCC",
        "His": "GTG", "Ile": "TAT", "Leu": "TAG", "Lys": "TTT",
        "Met": "TAC", "Phe": "GAA", "Pro": "TGG", "Ser": "AGT",
        "Thr": "TGT", "Trp": "CCA", "Tyr": "GTA", "Val": "CAT",
        "Ter": "TTA"  # Stop codon
    }
    alt_nucleotides: str = ""
    if (strand == 1):
        alt_nucleotides = protein_to_codon_p1.get(alt_long, "")
    elif (strand == -1):
        alt_nucleotides = protein_to_codon_n1.get(alt_long, "")
    else:
        raise ValueError(f"Invalid strand value: {strand}")
    if (ref_nucleotides == alt_nucleotides):
        raise ValueError(f"Reference and alternate nucleotides are the same: {ref_nucleotides}")
    return alt_nucleotides

def to_hgvs(raw_ensp: str, pos: int, alt_long: str) -> str:
    """
    Convert variant information to HGVS notation.
    Args:
        raw_ensp (str): Ensembl Protein ID (ENSP00000XXXXXX.X)
        pos (int): Position of the variant
        alt_long (str): Alternate amino acid (eg Leu for L/Leucine)
    Returns:
        str: HGVS notation (e.g., "ENSP00000354587.3:p.Pro175Leu")
    """
    ensp: str = raw_ensp.split(".")[0]  # Remove version number if present
    translation_response: requests.Response = requests.get(
        f"{ENSEMBL_API}/map/translation/{ensp}/{pos}..{pos}",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT
    )
    mappings_info: dict[str, Any] = translation_response.json().get("mappings", [{}])[0]
    seq_region_name: str = mappings_info.get("seq_region_name", "")
    start: int = mappings_info.get("start", 0)
    end: int = mappings_info.get("end", 0)
    strand: int = mappings_info.get("strand", 0)

    region_response: requests.Response = requests.get(
        f"{ENSEMBL_API}/info/region/{seq_region_name}",
        headers={"Content-Type": "text/plain"},
        timeout=TIMEOUT
    )
    ref_nucleotides: str = region_response.text.strip()

    nucleotide_change: str = get_nucleotide_change(ref_nucleotides, alt_long, strand)
    return f"{seq_region_name}:g.{start}_{end}delins{nucleotide_change}"

def get_vep_data(raw_ensp: str, pos: int, alt_long: str) -> dict:
    """
    Fetch variant effect prediction data from Ensembl VEP API.
    Assumes that hgvs is correctly formatted.
    Args:
        raw_ensp (str): Ensembl Protein ID (ENSP00000XXXXXX.X)
        pos (int): Position of the variant
        alt_long (str): Alternate amino acid (eg Leu for L/Leucine)
    Returns:
        dict: Parsed JSON response from the VEP API
    Raises:
        ValueError: If the API request fails or returns an error status code
    """
    hgvs = to_hgvs(raw_ensp, pos, alt_long)
    response: requests.Response = requests.get(
        f"{ENSEMBL_API}/vep/human/hgvs/{hgvs}",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT)
    if response.status_code != 200:
        raise ValueError(f"Error fetching VEP data for {hgvs}: {response.status_code}")
    return response.json()

def dict_to_pickle(dictionary: dict[Any, Any], file_location: os.PathLike) -> None:
    """
    Turns a dictionary into a pickle
    Args:
        dictionary (dict): dictionary that you want to pickle
        file_location (str): file location of where you want to dump the dictionary
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
