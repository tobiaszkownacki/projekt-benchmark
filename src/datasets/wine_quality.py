from typing import override

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset


class WineQualityDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        data_path_red = RAW_DATA_DIR / "winequality-red.csv"
        data_path_white = RAW_DATA_DIR / "winequality-white.csv"

        df_white = pd.read_csv(data_path_red, sep=";")
        df_red = pd.read_csv(data_path_white, sep=";")

        df_red["color"] = "red"
        df_white["color"] = "white"

        df_wines = pd.concat([df_red, df_white], ignore_index=True)
        df_wines = df_wines.sample(frac=1).reset_index(drop=True)

        y = df_wines["color"]
        y_binary = (y == "white").astype(int)
        X = df_wines.drop("color", axis=1)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_binary.astype(int).values, dtype=torch.long)

        full_dataset = TensorDataset(X_tensor, y_tensor)
        return ConcatDataset([full_dataset])
