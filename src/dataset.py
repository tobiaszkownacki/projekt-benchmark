"""
Code to download or generate data
"""

# from pathlib import Path
import pandas as pd
import torch
from src.config import RAW_DATA_DIR, ProblemType, DatasetType
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from models.cifar10 import Cifar10
from models.heart_disease import HeartDisease
from models.wine_quality import WineQuality
from models.digits import Digits
from models.abalone import Abalone


# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv"
#     # ----------------------------------------------
# ):

#     pass


# if __name__ == "__main__":
#     main()


class DataSetFactory:
    @classmethod
    def get_data_set(cls, data_set_name: str) -> Dataset:
        match data_set_name:
            case "cifar10":
                return cls._get_cifar10_data_set()
            case "heart_disease":
                return cls._get_heart_disease_data_set()
            case "wine_quality":
                return cls._get_wine_quality_data_set()
            case "digits":
                return cls._get_digits_data_set()
            case "abalone":
                return cls._get_abalone_data_set()
            case _:
                raise ValueError(f"Unsupported data set: {data_set_name}")

    @classmethod
    def _get_cifar10_data_set(cls) -> ConcatDataset:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            root=RAW_DATA_DIR / "cifar10",
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=RAW_DATA_DIR / "cifar10",
            train=True,
            download=True,
            transform=train_transforms,
        )
        return ConcatDataset([train_set, test_set])

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

        df_white = pd.read_csv(data_path_white, sep=";")
        df_red = pd.read_csv(data_path_red, sep=";")

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

    @classmethod
    def _get_digits_data_set(cls) -> ConcatDataset:
        digits = load_digits()
        X = digits.data
        y = digits.target

        X = X.reshape(-1, 1, 8, 8)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        X_tensor = X_tensor / 16.0  # digits data is originally in range [0, 16]

        full_dataset = TensorDataset(X_tensor, y_tensor)
        return full_dataset

    @classmethod
    def _get_abalone_data_set(cls) -> ConcatDataset:
        data_path = RAW_DATA_DIR / "abalone.data"
        df = pd.read_csv(data_path, sep=",", header=None)
        df[0] = df[0].map({"M": 0, "F": 1, "I": 2})

        X = df.drop(columns=[8])
        y = df[8]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        return TensorDataset(X_tensor, y_tensor)


DATA_SETS = {
    "cifar10": {
        "data_set": lambda: DataSetFactory.get_data_set("cifar10"),
        "supervised_model": Cifar10,
        "problem_type": ProblemType.CLASSIFICATION,
        "dataset_type": DatasetType.IMAGE,
        "input_shape": (3, 32, 32),
    },
    "heart_disease": {
        "data_set": lambda: DataSetFactory.get_data_set("heart_disease"),
        "supervised_model": HeartDisease,
        "problem_type": ProblemType.CLASSIFICATION,
        "dataset_type": DatasetType.TABULAR,
        "input_shape": (13,),
    },
    "wine_quality": {
        "data_set": lambda: DataSetFactory.get_data_set("wine_quality"),
        "supervised_model": WineQuality,
        "problem_type": ProblemType.CLASSIFICATION,
        "dataset_type": DatasetType.TABULAR,
        "input_shape": (12,),
    },
    "digits": {
        "data_set": lambda: DataSetFactory.get_data_set("digits"),
        "supervised_model": Digits,
        "problem_type": ProblemType.CLASSIFICATION,
        "dataset_type": DatasetType.IMAGE,
        "input_shape": (1, 8, 8),
    },
    "abalone": {
        "data_set": lambda: DataSetFactory.get_data_set("abalone"),
        "supervised_model": Abalone,
        "problem_type": ProblemType.REGRESSION,
        "dataset_type": DatasetType.TABULAR,
        "input_shape": (8,),
    },
}
