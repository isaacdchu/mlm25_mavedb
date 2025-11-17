"""
DataHandler module for loading and encoding data.
"""

from __future__ import annotations
from typing import List
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
from sklearn.preprocessing import MultiLabelBinarizer

class DataHandler:
    """
    Static class for handling data operations.
    """
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from a pickle file.
        Args:
            file_path (str): Path to the pickle file.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        with open(file_path, "rb") as f:
            data: pd.DataFrame = pickle.load(f)
        return data

    @staticmethod
    def one_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        One-hot encode specified columns in the DataFrame by adding new columns.
        Removes the original columns after encoding.
        Args:
            data (pd.DataFrame): The input DataFrame.
            columns (List[str]): List of column names to one-hot encode.
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
        Raises:
            KeyError: If any of the specified columns are not found in the DataFrame.
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
        Multi-hot encode specified columns in the DataFrame.
        Removes the original columns after encoding.
        Args:
            data (pd.DataFrame): The input DataFrame.
            columns (List[str]): List of column names to multi-hot encode.
        Returns:
            pd.DataFrame: DataFrame with multi-hot encoded columns.
        Raises:
            KeyError: If any of the specified columns are not found in the DataFrame.
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
        return new_data
