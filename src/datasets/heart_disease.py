from typing import override

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset


class HeartDiseaseDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        data_path = RAW_DATA_DIR / "heart+disease.data"
        df = pd.read_csv(data_path, sep=";", header=0, na_values=["?"])
        num_classes = df["target"].nunique()

        df = df.dropna().copy()

        if df["target"].dtype in ["float64", "int64"]:
            df["target"] = pd.cut(
                df["target"], bins=num_classes, labels=range(num_classes)
            )

        X = df.drop(columns=["target"])
        y = df["target"]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])
