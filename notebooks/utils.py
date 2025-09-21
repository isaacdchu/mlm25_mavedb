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

# Protein-nucleotide mapping (codons)
# Note: Each amino acid can be coded by multiple codons; here we list all possible codons
# Strand 1 (positive strand)
PROTEIN_TO_NUCLEOTIDES_P1: dict[str, list[str]] = {
    "Ala": ["GCA", "GCC", "GCG", "GCT"],
    "Arg": ["AGA", "AGG", "CGA", "CGC", "CGG", "CGT"],
    "Asn": ["AAC", "AAT"],
    "Asp": ["GAC", "GAT"],
    "Cys": ["TGC", "TGT"],
    "Gln": ["CAA", "CAG"],
    "Glu": ["GAA", "GAG"],
    "Gly": ["GGA", "GGC", "GGG", "GGT"],
    "His": ["CAC", "CAT"],
    "Ile": ["ATA", "ATC", "ATT"],
    "Leu": ["CTA", "CTC", "CTG", "CTT", "TTA", "TTG"],
    "Lys": ["AAA", "AAG"],
    "Met": ["ATG"],
    "Phe": ["TTC", "TTT"],
    "Pro": ["CCA", "CCC", "CCG", "CCT"],
    "Ser": ["AGC", "AGT", "TCA", "TCC", "TCG", "TCT"],
    "Thr": ["ACA", "ACC", "ACG", "ACT"],
    "Trp": ["TGG"],
    "Tyr": ["TAC", "TAT"],
    "Val": ["GTA", "GTC", "GTG", "GTT"],
    "Ter": ["TAA", "TAG", "TGA"]  # Stop codons
}

# Strand -1 (reverse complement)
def reverse_complement(dna_seq: str) -> str:
    """
    Returns the reverse complement of a DNA sequence.
    Args:
        dna_seq (str): Input DNA sequence
    Returns:
        str: Reverse complement of the input sequence
    """
    complement = str.maketrans("ACGT", "TGCA")
    return dna_seq.translate(complement)[::-1]

PROTEIN_TO_NUCLEOTIDES_N1: dict[str, list[str]] = {
    aa: [reverse_complement(codon) for codon in codons]
    for aa, codons in PROTEIN_TO_NUCLEOTIDES_P1.items()
}

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

def hamming_distance(s1: str, s2: str) -> int:
    """
    Calculate the Hamming distance between two strings.
    Assumes that the strings are of equal length.
    Args:
        s1 (str): First string
        s2 (str): Second string
    Returns:
        int: Hamming distance
    """
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def find_best_substitution(ref_nucleotides: str, alt_nucleotides_candidates: list[str]) -> str:
    """
    Finds the best matching nucleotide substitution from a list of candidates.
    (Best match is lowest hamming distance, ties broken arbitrarily)
    Args:
        ref_nucleotides (str): Reference nucleotide sequence (eg. AAG)
        alt_nucleotides_candidates (list[str]): List of candidate alternate nucleotide sequences
    Returns:
        str: The best matching alternate nucleotide sequence (eg. AAC)
    """
    best_candidate: str = ""
    for candidate in alt_nucleotides_candidates:
        distance: int = hamming_distance(ref_nucleotides, candidate)
        if (distance > 0):
            return candidate
        else:
            best_candidate = candidate
    if (best_candidate != ""):
        print(f"Warning: No nucleotide change found for {ref_nucleotides}, using {best_candidate} from candidates: {alt_nucleotides_candidates}")
        return best_candidate
    raise ValueError(f"No valid nucleotide substitution found for {ref_nucleotides} from candidates: {alt_nucleotides_candidates}")
    

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
    alt_nucleotides_candidates: list[str] = []
    
    # Strand is 1: use normal codons
    if (strand == 1):
        alt_nucleotides_candidates = PROTEIN_TO_NUCLEOTIDES_P1.get(alt_long, [""])
    # Strand is -1: use reverse complement codons
    elif (strand == -1):
        alt_nucleotides_candidates = PROTEIN_TO_NUCLEOTIDES_N1.get(alt_long, [""])
    else:
        raise ValueError(f"Invalid strand value: {strand}")

    alt_nucleotides: str = find_best_substitution(ref_nucleotides, alt_nucleotides_candidates)
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
    strand: int = mappings_info.get("strand", 0)
    start: int = mappings_info.get("start", -3)
    end: int = start + 2

    region_response: requests.Response = requests.get(
        f"{ENSEMBL_API}/sequence/region/human/{seq_region_name}:{start}..{end}:{strand}?",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT
    )
    ref_nucleotides: str = region_response.json().get("seq", "")
    alt_nucleotides: str = get_nucleotide_change(ref_nucleotides, alt_long, strand)
    print(f"ENSP: {raw_ensp}, Pos: {pos}, Ref: {ref_nucleotides}, Alt: {alt_long}:{alt_nucleotides}, Strand: {strand}")
    return f"{seq_region_name}:g.{start}_{end}delins{alt_nucleotides}"

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
        raise ValueError(f"Error fetching VEP data for {hgvs}: {response.status_code}\n"
                         f"Response: {response.text}")
    return response.json()[0]

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
