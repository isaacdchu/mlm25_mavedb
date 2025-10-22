"""
Inputs:
1. Processed CSV file
2. Full sequence mappings
3. VEP data mappings
4. ESM C embeddings
"""

import os
from pathlib import Path
import pickle
import yaml
import pandas as pd
import torch

class DataHandler:
    """
    The class will allow for easy access to train data
    Class will be fully static methods
    """
    train_path: Path = Path("")
    train_vep_path: Path = Path("")
    train_embeddings_map_path: Path = Path("")
    sequence_map_path: Path = Path("")
    train_vep: dict[str, dict] = {}
    embeddings_map: dict[str, torch.Tensor] = {}
    sequence_map: dict[str, str] = {}

    # Setup config
    def __init__(self, config_path: str, data_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            DataHandler.config = yaml.safe_load(f)

        # Build paths robustly (strip leading slashes from config entries)
        base = Path(data_path)
        DataHandler.train_path = (base / DataHandler.config["data"]["train_path"].lstrip("/")).resolve()
        DataHandler.train_vep_path = (base / DataHandler.config["data"]["train_vep_path"].lstrip("/")).resolve()
        DataHandler.train_embeddings_map_path = (base / DataHandler.config["data"]["train_embeddings_map_path"].lstrip("/")).resolve()
        DataHandler.sequence_map_path = (base / DataHandler.config["data"]["sequence_map_path"].lstrip("/")).resolve()

        # Helpful debug if files are missing
        for name, p in [
            ("train_path", DataHandler.train_path),
            ("train_vep_path", DataHandler.train_vep_path),
            ("train_embeddings_map_path", DataHandler.train_embeddings_map_path),
            ("sequence_map_path", DataHandler.sequence_map_path),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{name} not found at {p}")

        # Load mappings
        with open(DataHandler.train_vep_path, "rb") as f:
            DataHandler.train_vep = pickle.load(f)
        with open(DataHandler.train_embeddings_map_path, "rb") as f:
            DataHandler.embeddings_map = pickle.load(f)
        with open(DataHandler.sequence_map_path, "rb") as f:
            DataHandler.sequence_map = pickle.load(f)

    @staticmethod
    def train_next_line():
        """
        Generator to read the training data one line at a time
        Yields:
            pd.DataFrame: A single row DataFrame from the training data
        """
        for chunk in pd.read_csv(DataHandler.train_path, chunksize=1):
            yield chunk

    @staticmethod
    def get_vep(ensp: str, position: int, ref_long: str, alt_long: str) -> dict | None:
        """
        Get VEP data for a given row of data
        Args:
            ensp (str): Ensembl protein ID
            position (int): Position of the variant
            ref_long (str): Reference amino acid
            alt_long (str): Alternate amino acid
        Returns:
            dict | None: VEP data dictionary or None if not found
        """
        key: str = f"{ensp}_{position}_{ref_long}_{alt_long}"
        return DataHandler.train_vep.get(key, None)

    @staticmethod
    def get_embedding(ensp: str, position: int, ref_long: str, alt_long: str) -> torch.Tensor | None:
        """
        Get ESM C embedding difference for a given row of data
        Args:
            ensp (str): Ensembl protein ID
            position (int): Position of the variant
            ref_long (str): Reference amino acid
            alt_long (str): Alternate amino acid
        Returns:
            torch.Tensor | None: ESM C embedding difference tensor or None if not found
        """
        key: str = f"{ensp}_{position}_{ref_long}_{alt_long}"
        return DataHandler.embeddings_map.get(key, None)

    @staticmethod
    def get_sequence(ensp: str) -> str | None:
        """
        Get full protein sequence for a given Ensembl protein ID
        Args:
            ensp (str): Ensembl protein ID
        Returns:
            str | None: Full protein sequence or None if not found
        """
        return DataHandler.sequence_map.get(ensp, None)
