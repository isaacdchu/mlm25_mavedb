from __future__ import annotations
from typing import List
import pandas as pd

class DataHandler:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        pass

    @staticmethod
    def one_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass

    @staticmethod
    def multi_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass
