"""
Code to download or generate data
"""

from torch.utils.data import Dataset

from models.cifar10 import Cifar10
from models.digits import Digits
from models.heart_disease import HeartDisease
from models.light.students_performance_light import StudentsPerformanceLight
from models.medium.students_performance_medium import StudentsPerformanceMedium
from models.heavy.students_performance_heavy import StudentsPerformanceHeavy
from models.wine_quality import WineQuality
from models.digits_mlp import DigitsMLP

from src.datasets.cifar10 import Cifar10Dataset
from src.datasets.digits import DigitsDataset
from src.datasets.heart_disease import HeartDiseaseDataset
from src.datasets.wine_quality import WineQualityDataset
from src.datasets.students_performance import StudentsPerformanceDataset


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
            case "students_performance":
                return StudentsPerformanceDataset().get()
            case _:
                raise ValueError(f"Unsupported data set: {data_set_name}")


DATA_SETS = {
    "cifar10": {
        "data_set": lambda: DataSetFactory.get_data_set("cifar10"),
    },
    "heart_disease": {
        "data_set": lambda: DataSetFactory.get_data_set("heart_disease"),
    },
    "wine_quality": {
        "data_set": lambda: DataSetFactory.get_data_set("wine_quality"),
    },
    "digits": {
        "data_set": lambda: DataSetFactory.get_data_set("digits"),
    },
    "students_performance": {
        "data_set": lambda: DataSetFactory.get_data_set("students_performance")
    }
}


MODELS = {
    "cifar10": {
        "default": Cifar10,
        # "resnet": Cifar10ResNet,
    },
    "heart_disease": {
        "default": HeartDisease,
    },
    "wine_quality": {
        "default": WineQuality,
    },
    "digits": {
        "default": Digits,
        "mlp": DigitsMLP,
    },

    "students_performance": {
        "light": StudentsPerformanceLight,
        "medium": StudentsPerformanceMedium,
        "heavy": StudentsPerformanceHeavy,
    }
}
