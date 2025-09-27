'''
Utility functions and constants for data processing and API interactions.
'''

import os
from pathlib import Path
import pickle
from typing import Any
import requests

from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    LogitsConfig,
    LogitsOutput,
)
from torch import Tensor
import torch

# All file paths
RAW_TRAIN_PATH: os.PathLike = Path("../data/train/raw_train.csv")
RAW_TEST_PATH: os.PathLike = Path("../data/test/raw_test.csv")
PROCESSED_TEST_PATH: os.PathLike = Path("../data/test/processed_test.csv")
PROCESSED_TRAIN_PATH: os.PathLike = Path("../data/train/processed_train.csv")
ENSP_SEQUENCE_MAP_PATH: os.PathLike = Path("../data/ensp_sequence_map.pkl")
TRAIN_VEP_DATA_PATH: os.PathLike = Path("../data/train/train_vep.pkl")

ENSP_EMBEDDINGS_MAP_PATH: os.PathLike = Path("../data/ensp_embeddings_map.pkl")

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
PROTEIN_TO_NUCLEOTIDES_N1: dict[str, list[str]] = {
    "Ala": ["TGC", "GGC", "CGC", "AGC"],
    "Arg": ["TCT", "TCC", "GCT", "GGC", "GCT", "ACC"],
    "Asn": ["GTT", "GTT"],
    "Asp": ["GTC", "GTC"],
    "Cys": ["GCA", "GCA"],
    "Gln": ["TTG", "CTG"],
    "Glu": ["TTC", "CTC"],
    "Gly": ["TCC", "GCC", "CCC", "ACC"],
    "His": ["GTG", "GTG"],
    "Ile": ["TAT", "GAT", "AAT"],
    "Leu": ["TAG", "CAG", "CAG", "CAG", "TAA", "CAA"],
    "Lys": ["TTT", "CTT"],
    "Met": ["CAT"],
    "Phe": ["GAA", "GAA"],
    "Pro": ["TGG", "GGG", "CCG", "AGG"],
    "Ser": ["TCT", "TCC", "GCT", "GGC", "CCG", "AGC"],
    "Thr": ["TGA", "GGA", "CGA", "ACA"],
    "Trp": ["CCA"],
    "Tyr": ["GTA", "ATA"],
    "Val": ["CAT", "GTA", "CAC", "AAC"],
    "Ter": ["TTA", "CTA", "TCA"]  # Stop codons
}

# ESM C Model
N_KMEANS_CLUSTERS: int = 3
EMBEDDING_CONFIG: LogitsConfig = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE: str = "esmc_600m"
ESM_MODEL: ESMC = ESMC.from_pretrained(MODEL_TYPE).to(DEVICE) # cuda or cpu

# Helper functions
# Full sequence
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

def sequence_substitution(sequence: str, pos: int, alt_short: str) -> str:
    """
    Substitute the amino acid at the given position in the sequence with the alternate amino acid.
    Assumes valid inputs.
    Args:
        sequence (str): Original amino acid sequence
        pos (int): Position to substitute (1-indexed)
        alt_short (str): Alternate amino acid (single-letter code)
    Returns:
        str: Modified amino acid sequence
    """
    if (alt_short == "*"):  # Stop codon
        return sequence[:pos-1]  # Truncate sequence at position
    return sequence[:pos-1] + alt_short + sequence[pos:]

# VEP data
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
        str: HGVS notation (e.g., "X:g.15594962..15594964delinsTTA")
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
    return f"{seq_region_name}:g.{start}..{end}delins{alt_nucleotides}"

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

# ESM C embedding utilities
def get_embedding(sequence: str) -> Tensor:
    """
    Get the ESM C embedding for a given protein sequence.
    Args:
        sequence (str): Protein sequence
    Returns:
        Tensor: The output embeddings tensor of shape (sequence_length, 1152)
    Raises:
        ValueError: If the embeddings are not found in the logits output
    """
    # Handle sequences longer than 2048 by splitting into chunks
    if len(sequence) > 2046:
        # Take average of 2 embeddings of chunks
        protein_1: ESMProtein = ESMProtein(sequence=sequence[:2046])
        protein_2: ESMProtein = ESMProtein(sequence=sequence[-2046:])
        protein_tensor_1: ESMProteinTensor = ESM_MODEL.encode(protein_1)
        protein_tensor_2: ESMProteinTensor = ESM_MODEL.encode(protein_2)
        logits_1: LogitsOutput = ESM_MODEL.logits(protein_tensor_1, EMBEDDING_CONFIG)
        logits_2: LogitsOutput = ESM_MODEL.logits(protein_tensor_2, EMBEDDING_CONFIG)
        output_1: Tensor | None = logits_1.embeddings
        output_2: Tensor | None = logits_2.embeddings
        if output_1 is None or output_2 is None:
            raise ValueError(f"Embeddings not found in logits output for sequence: {sequence}")
        output_1 = output_1.squeeze(0)[1:-1]  # Remove start/end tokens
        output_2 = output_2.squeeze(0)[1:-1]  # Remove start/end tokens
        # Overlap the proteins on their intersection
        overlap_amount: int = 2 * 2046 - len(sequence)
        # Get overlapping slices
        overlap_1 = output_1[-overlap_amount:]
        overlap_2 = output_2[:overlap_amount]
        # Average the overlapping area
        overlap = torch.mean(torch.stack([overlap_1, overlap_2]), dim=0)
        # Concatenate the non-overlapping parts with the averaged overlap
        joint_output: Tensor = torch.cat((output_1[:-overlap_amount], overlap, output_2[overlap_amount:]), dim=0)
        return joint_output.mean(dim=0)  # Mean over sequence length

    # Sequence is within limit
    protein: ESMProtein = ESMProtein(sequence=sequence)
    protein_tensor: ESMProteinTensor = ESM_MODEL.encode(protein)
    logits: LogitsOutput = ESM_MODEL.logits(protein_tensor, EMBEDDING_CONFIG)
    output: Tensor | None = logits.embeddings
    if output is None:
        raise ValueError(f"Embeddings not found in logits output for sequence: {sequence}")
    # Return the first (and only) element in the batch and remove start/end tokens
    return output.squeeze(0)[1:-1].mean(dim=0)  # Mean over sequence length

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

def vep_from_pickle(vep_data_path: os.PathLike) -> list[dict]:
    """
    Load VEP data from a pickle file.
    Args:
        vep_data_path (os.PathLike): Path to the pickle file containing VEP data
    Returns:
        list[dict]: List of dictionaries containing VEP data
    """
    vep_data: list[dict] = []
    with open(vep_data_path, "rb") as f:
        while True:
            try:
                vep_data.append(pickle.load(f))
            except EOFError:
                break
    return vep_data

def save_embeddings(embeddings: list[LogitsOutput], path: os.PathLike) -> None:
    """
    Save a list of LogitsOutput embeddings to a pickle file.
    Args:
        embeddings (list[LogitsOutput]): List of LogitsOutput objects to save
        path (os.PathLike): Path to the output pickle file
    Returns:
        None
    """
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings(path: os.PathLike) -> list[LogitsOutput]:
    """
    Load a list of LogitsOutput embeddings from a pickle file.
    Args:
        path (os.PathLike): Path to the input pickle file
    Returns:
        list[LogitsOutput]: List of LogitsOutput objects loaded from the file
    """
    with open(path, "rb") as f:
        embeddings: list[LogitsOutput] = pickle.load(f)
    return embeddings
