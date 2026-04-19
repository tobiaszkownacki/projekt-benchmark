"""
Code to download or generate data
"""

from torch.utils.data import Dataset

from models.cifar10 import Cifar10
from models.digits import Digits
from models.heart_disease import HeartDisease
from models.wine_quality import WineQuality
from src.datasets.cifar10 import Cifar10Dataset
from src.datasets.digits import DigitsDataset
from src.datasets.heart_disease import HeartDiseaseDataset
from src.datasets.wine_quality import WineQualityDataset


class DataSetFactory:
    @classmethod
    def get_data_set(cls, data_set_name: str) -> Dataset:
        match data_set_name:
            case "cifar10":
                return Cifar10Dataset().get()
            case "heart_disease":
                return HeartDiseaseDataset().get()
            case "wine_quality":
                return WineQualityDataset().get()
            case "digits":
                return DigitsDataset().get()
            case _:
                raise ValueError(f"Unsupported data set: {data_set_name}")


DATA_SETS = {
    "cifar10": {
        "data_set": lambda: DataSetFactory.get_data_set("cifar10"),
        "model": Cifar10,
    },
    "heart_disease": {
        "data_set": lambda: DataSetFactory.get_data_set("heart_disease"),
        "model": HeartDisease,
    },
    "wine_quality": {
        "data_set": lambda: DataSetFactory.get_data_set("wine_quality"),
        "model": WineQuality,
    },
    "digits": {
        "data_set": lambda: DataSetFactory.get_data_set("digits"),
        "model": Digits,
    },
}
