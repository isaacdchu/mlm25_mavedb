"""
DataHandler module for loading and encoding data
"""

from __future__ import annotations
from typing import List, Any, Callable, Dict, Optional, Tuple
import pickle
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from scipy.sparse import spmatrix
from sklearn.preprocessing import MultiLabelBinarizer


class DataHandler:
    """
    Static class for handling data operations
    """

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from a pickle file
        Args:
            file_path (str): Path to the pickle file
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        with open(file_path, "rb") as f:
            data: pd.DataFrame = pickle.load(f)
        return data

    @staticmethod
    def one_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        One-hot encode specified columns in the DataFrame by adding new columns
        Removes the original columns after encoding
        Args:
            data (pd.DataFrame): The input DataFrame
            columns (List[str]): List of column names to one-hot encode
        Returns: pd.DataFrame
            DataFrame with one-hot encoded columns
        Raises:
            KeyError:
                If any of the specified columns are not found in the DataFrame.
        """
        # validate columns
        missing = set(columns) - set(data.columns)
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {sorted(missing)}")
        new_df: pd.DataFrame = pd.get_dummies(data, columns=columns, prefix=columns, dtype=int)
        return new_df

    @staticmethod
    def multi_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Multi-hot encode specified columns in the DataFrame
        Removes the original columns after encoding
        Args:
            data (pd.DataFrame): The input DataFrame
            columns (List[str]): List of column names to multi-hot encode
        Returns: pd.DataFrame
            DataFrame with multi-hot encoded columns
        Raises:
            KeyError:
                If any of the specified columns are not found in the DataFrame
        """
        # validate columns
        missing = set(columns) - set(data.columns)
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {sorted(missing)}")
        # use a sparse MultiLabelBinarizer for memory efficiency on large data
        new_data: pd.DataFrame = data.copy(deep=True)
        for col in columns:
            # normalize each cell to an iterable of labels (treat NaN as empty)
            col_vals: pd.Series = new_data[col].apply(
                lambda x:
                    [] if any(pd.isna(x))
                    else (list(x) if isinstance(x, (list, tuple, set)) else [x])
            )
            mlb: MultiLabelBinarizer = MultiLabelBinarizer(sparse_output=True)
            mat: np.ndarray | spmatrix = mlb.fit_transform(col_vals)
            if mat.shape[1] == 0:
                continue
            col_names: List[str] = [f"{col}_{c}" for c in mlb.classes_]
            dummies: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(
                data=mat,
                index=new_data.index,
                columns=col_names
            )
            new_data = new_data.join(dummies)
        new_data.drop(columns=columns, inplace=True)
        # densify to avoid downstream sparse masking issues
        return new_data.sparse.to_dense() if hasattr(new_data, "sparse") else new_data

    @staticmethod
    def add_rel_pos(
        data: pd.DataFrame,
        sequence_length_map: Optional[Dict[str, int]] = None,
        pos_col: str = "pos",
        seq_len_col: str = "sequence_length",
        seq_col: str = "sequence"
    ) -> pd.DataFrame:
        """
        Ensure a rel_pos column is present: rel_pos = pos / sequence_length.
        Uses sequence_length column, sequence string length, or a provided map keyed by "ensp".
        """
        new_data: pd.DataFrame = data.copy(deep=True)
        if "rel_pos" in new_data.columns or pos_col not in new_data.columns:
            return new_data

        pos_vals: pd.Series = pd.to_numeric(new_data[pos_col], errors="coerce")
        length_vals: Optional[pd.Series] = None
        if seq_len_col in new_data.columns:
            length_vals = pd.to_numeric(new_data[seq_len_col], errors="coerce")
        elif seq_col in new_data.columns:
            length_vals = new_data[seq_col].apply(lambda s: float(len(s)) if isinstance(s, str) else np.nan)
        elif sequence_length_map is not None and "ensp" in new_data.columns:
            length_vals = new_data["ensp"].map(sequence_length_map).astype(float)

        if length_vals is None:
            return new_data

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_pos: pd.Series = pos_vals / length_vals.replace(0, np.nan)
        new_data["rel_pos"] = rel_pos
        return new_data

    @staticmethod
    def compute_embedding_diff(
        data: pd.DataFrame,
        ref_col: str = "ref_embedding",
        alt_col: str = "alt_embedding",
        out_col: str = "emb_diff"
    ) -> pd.DataFrame:
        """
        Compute emb_diff = alt_embedding - ref_embedding when both are present.
        Leaves the data unchanged if embeddings are missing.
        """
        if ref_col not in data.columns or alt_col not in data.columns:
            return data.copy(deep=True)
        new_data: pd.DataFrame = data.copy(deep=True)

        def _diff(row: pd.Series) -> Tensor:
            ref = torch.as_tensor(row[ref_col], dtype=torch.float32)
            alt = torch.as_tensor(row[alt_col], dtype=torch.float32)
            return torch.nan_to_num(alt - ref)

        new_data[out_col] = new_data.apply(_diff, axis=1)
        return new_data

    @staticmethod
    def encode_scoreset_id(
        data: pd.DataFrame,
        scoreset_col: str = "scoreset",
        out_col: str = "scoreset_id",
        scoreset_to_id: Optional[Dict[str, int]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Add a numeric scoreset_id column. If a mapping is provided, reuse it; otherwise create one.
        Returns the updated DataFrame and the mapping used.
        """
        new_data: pd.DataFrame = data.copy(deep=True)
        mapping: Dict[str, int] = scoreset_to_id or {}
        if scoreset_col not in new_data.columns:
            return new_data, mapping

        if not mapping:
            unique_scoresets: List[str] = sorted(new_data[scoreset_col].unique())
            mapping = {name: idx for idx, name in enumerate(unique_scoresets)}
        new_data[out_col] = new_data[scoreset_col].map(mapping)
        return new_data, mapping

    @staticmethod
    def build_feature_set_a(
        data: pd.DataFrame,
        sequence_length_map: Optional[Dict[str, int]] = None,
        cat_cols: Optional[List[str]] = None,
        include_scoreset_id: bool = False
    ) -> pd.DataFrame:
        """
        Feature Set A: tabular-only (rel_pos + one-hot categorical).
        Removes embeddings and optional identifier columns while preserving the score column.
        """
        new_data: pd.DataFrame = DataHandler.add_rel_pos(data, sequence_length_map=sequence_length_map)

        default_cat: List[str] = [
            "ref_long",
            "alt_long",
            "ref_short",
            "alt_short",
            "consequence",
            "biotype",
        ]
        if not include_scoreset_id:
            default_cat.append("scoreset")
        chosen_cats: List[str] = cat_cols if cat_cols is not None else [c for c in default_cat if c in new_data.columns]
        if chosen_cats:
            new_data = DataHandler.one_hot_encode(new_data, columns=chosen_cats)

        if include_scoreset_id:
            new_data, _ = DataHandler.encode_scoreset_id(new_data)
            if "scoreset" in new_data.columns:
                new_data = new_data.drop(columns=["scoreset"])

        drop_cols: List[str] = [col for col in [
            "accession",
            "sequence",
            "sequence_length",
            "ref_embedding",
            "alt_embedding",
            "emb_diff"
        ] if col in new_data.columns]
        if "rel_pos" in new_data.columns and "pos" in new_data.columns:
            drop_cols.append("pos")
        if drop_cols:
            new_data = new_data.drop(columns=drop_cols)

        # Reorder to keep score first when present
        if "score" in new_data.columns:
            feature_cols = [c for c in new_data.columns if c != "score"]
            new_data = new_data[["score"] + feature_cols]
        return new_data

    @staticmethod
    def build_feature_set_b(
        data: pd.DataFrame,
        sequence_length_map: Optional[Dict[str, int]] = None,
        cat_cols: Optional[List[str]] = None,
        include_scoreset_id: bool = False
    ) -> pd.DataFrame:
        """
        Feature Set B: tabular features plus embedding difference.
        """
        new_data: pd.DataFrame = DataHandler.add_rel_pos(data, sequence_length_map=sequence_length_map)
        new_data = DataHandler.compute_embedding_diff(new_data)

        default_cat: List[str] = [
            "ref_long",
            "alt_long",
            "ref_short",
            "alt_short",
            "consequence",
            "biotype",
        ]
        if not include_scoreset_id:
            default_cat.append("scoreset")
        chosen_cats: List[str] = cat_cols if cat_cols is not None else [c for c in default_cat if c in new_data.columns]
        if chosen_cats:
            new_data = DataHandler.one_hot_encode(new_data, columns=chosen_cats)

        if include_scoreset_id:
            new_data, _ = DataHandler.encode_scoreset_id(new_data)
            if "scoreset" in new_data.columns:
                new_data = new_data.drop(columns=["scoreset"])

        drop_cols: List[str] = [col for col in [
            "accession",
            "sequence",
            "sequence_length",
            "ref_embedding",
            "alt_embedding"
        ] if col in new_data.columns]
        if "rel_pos" in new_data.columns and "pos" in new_data.columns:
            drop_cols.append("pos")
        if drop_cols:
            new_data = new_data.drop(columns=drop_cols)

        if "score" in new_data.columns:
            feature_cols = [c for c in new_data.columns if c != "score"]
            new_data = new_data[["score"] + feature_cols]
        return new_data

    @staticmethod
    def build_feature_set_c(data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Set C: embeddings-only via emb_diff.
        """
        new_data: pd.DataFrame = DataHandler.compute_embedding_diff(data)
        if "emb_diff" not in new_data.columns:
            raise KeyError("emb_diff could not be constructed because embeddings are missing.")
        keep_cols: List[str] = ["score", "emb_diff"] if "score" in new_data.columns else ["emb_diff"]
        return new_data[keep_cols]

    # @staticmethod
    # def embed(data: pd.DataFrame, columns: List[str], embed_func: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    #     """
    #     Embed specified columns in the DataFrame using a provided embedding function
    #     Args:
    #         data (pd.DataFrame): The input DataFrame
    #         columns (List[str]): List of column names to embed
    #         embed_func (Callable[[Any], None]): Function that takes a value and returns its embedding as a numpy array
    #     Returns: pd.DataFrame
    #         DataFrame with embedded columns
    #     Raises:
    #         KeyError:
    #             If any of the specified columns are not found in the DataFrame
    #     """
    #     # validate columns
    #     missing = set(columns) - set(data.columns)
    #     if missing:
    #         raise KeyError(f"Columns not found in DataFrame: {sorted(missing)}")
    #     new_data: pd.DataFrame = data.copy(deep=True)
    #     for col in columns:
    #         embeddings: pd.Series = embed_func(new_data[col])
    #         new_data[col] = embeddings
    #     return new_data
