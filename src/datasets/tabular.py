import pandas as pd
import torch
from torch.utils.data import ConcatDataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR
from src.datasets.base import BaseDatasetFactory


class TabularDatasetFactory(BaseDatasetFactory):
    """Factory for tabular datasets"""
    
    @classmethod
    def get_data_set(cls, dataset_name: str) -> ConcatDataset:
        match dataset_name:
            case "heart_disease":
                return cls._get_heart_disease_data_set()
            case "wine_quality":
                return cls._get_wine_quality_data_set()
            case _:
                raise ValueError(f"Unsupported tabular dataset: {dataset_name}")
    
    @classmethod
    def get_supported_datasets(cls) -> list[str]:
        return ["heart_disease", "wine_quality"]
    
    @classmethod
    def _get_heart_disease_data_set(cls) -> ConcatDataset:
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

        return train_dataset
    
    @classmethod
    def _get_wine_quality_data_set(cls) -> ConcatDataset:
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
        return full_dataset
