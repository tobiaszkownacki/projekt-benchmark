"""
Code to download or generate data
"""

# from pathlib import Path
import pandas as pd
import torch
from src.config import RAW_DATA_DIR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from models.cifar10 import Cifar10
from models.heart_disease import HeartDisease
from models.wine_quality import WineQuality
from models.digits import Digits


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
            case 'cifar10':
                return cls._get_cifar10_data_set()
            case 'heart_disease':
                return cls._get_heart_disease_data_set()
            case 'wine_quality':
                return cls._get_wine_quality_data_set()
            case 'digits':
                return cls._get_digits_data_set()
            case _:
                raise ValueError(f'Unsupported data set: {data_set_name}')

    @classmethod
    def _get_cifar10_data_set(cls) -> ConcatDataset:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop(32, padding=4)])

        train_set = torchvision.datasets.CIFAR10(
            root=RAW_DATA_DIR / 'cifar10',
            train=True,
            download=True,
            transform=train_transforms
        )
        test_set = torchvision.datasets.CIFAR10(
            root=RAW_DATA_DIR / 'cifar10',
            train=True,
            download=True,
            transform=train_transforms
        )
        return ConcatDataset([train_set, test_set])

    @classmethod
    def _get_heart_disease_data_set(cls) -> ConcatDataset:
        data_path = RAW_DATA_DIR / "heart+disease.data"
        df = pd.read_csv(data_path, sep=";", header=0, na_values=["?"])
        num_classes = df["target"].nunique()

        df = df.dropna().copy()

        if df["target"].dtype in ["float64", "int64"]:
            df["target"] = pd.cut(df["target"], bins=num_classes, labels=range(num_classes))

        X = df.drop(columns=["target"])
        y = df["target"]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        full_dataset = TensorDataset(X_tensor, y_tensor)

        return full_dataset

    @classmethod
    def _get_wine_quality_data_set(cls) -> ConcatDataset:
        data_path_red = RAW_DATA_DIR / "winequality-red.csv"
        data_path_white = RAW_DATA_DIR / "winequality-white.csv"

        df_white = pd.read_csv(data_path_red, sep=';')
        df_red = pd.read_csv(data_path_white, sep=';')

        df_red['color'] = 'red'
        df_white['color'] = 'white'

        df_wines = pd.concat([df_red, df_white], ignore_index=True)
        df_wines = df_wines.sample(frac=1).reset_index(drop=True)

        y = df_wines['color']
        y_binary = (y == 'white').astype(int)
        X = df_wines.drop('color', axis=1)

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


DATA_SETS = {
    'cifar10': {
        "data_set": lambda: DataSetFactory.get_data_set('cifar10'),
        "model": Cifar10
    },
    'heart_disease': {
        "data_set": lambda: DataSetFactory.get_data_set('heart_disease'),
        "model": HeartDisease
    },
    'wine_quality': {
        "data_set": lambda: DataSetFactory.get_data_set('wine_quality'),
        "model": WineQuality
    },
    'digits': {
        "data_set": lambda: DataSetFactory.get_data_set('digits'),
        "model": Digits
    }
}
