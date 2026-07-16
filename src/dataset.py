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
from models.light.fashion_mnist_light import FashionMNISTLight
from models.medium.fashion_mnist_medium import FashionMNISTMedium
from models.heavy.fashion_mnist_heavy import FashionMNISTHeavy
from models.light.stl10_light import STL10Light
from models.medium.stl10_medium import STL10Medium
from models.heavy.stl10_heavy import STL10Heavy
from models.light.mobile_price_light import MobilePriceLight
from models.medium.mobile_price_medium import MobilePriceMedium
from models.heavy.mobile_price_heavy import MobilePriceHeavy
from models.light.healthcare_light import HealthcareLight
from models.medium.healthcare_medium import HealthcareMedium
from models.heavy.healthcare_heavy import HealthcareHeavy
from models.light.apple_quality_light import AppleQualityLight
from models.medium.apple_quality_medium import AppleQualityMedium
from models.heavy.apple_quality_heavy import AppleQualityHeavy
from models.wine_quality import WineQuality
from models.digits_mlp import DigitsMLP

from src.datasets.cifar10 import Cifar10Dataset
from src.datasets.digits import DigitsDataset
from src.datasets.heart_disease import HeartDiseaseDataset
from src.datasets.wine_quality import WineQualityDataset
from src.datasets.students_performance import StudentsPerformanceDataset
from src.datasets.fashion_mnist import FashionMNISTDataset
from src.datasets.stl10 import STL10Dataset
from src.datasets.mobile_price import MobilePriceDataset
from src.datasets.healthcare import HealthcareDataset
from src.datasets.apple_quality import AppleQualityDataset


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
            case "fashion_mnist":
                return FashionMNISTDataset().get()
            case "stl10":
                return STL10Dataset().get()
            case "mobile_price":
                return MobilePriceDataset().get()
            case "healthcare":
                return HealthcareDataset().get()
            case "apple_quality":
                return AppleQualityDataset().get()
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
    },
    "fashion_mnist": {
        "data_set": lambda: DataSetFactory.get_data_set("fashion_mnist")
    },
    "stl10": {
        "data_set": lambda: DataSetFactory.get_data_set("stl10")
    },
    "mobile_price": {
        "data_set": lambda: DataSetFactory.get_data_set("mobile_price")
    },
    "healthcare": {
        "data_set": lambda: DataSetFactory.get_data_set("healthcare")
    },
    "apple_quality": {
        "data_set": lambda: DataSetFactory.get_data_set("apple_quality")
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
    },

    "fashion_mnist": {
        "light": FashionMNISTLight,
        "medium": FashionMNISTMedium,
        "heavy": FashionMNISTHeavy,
    },

    "stl10": {
        "light": STL10Light,
        "medium": STL10Medium,
        "heavy": STL10Heavy,
    },

    "mobile_price": {
        "light": MobilePriceLight,
        "medium": MobilePriceMedium,
        "heavy": MobilePriceHeavy,
    },

    "healthcare": {
        "light": HealthcareLight,
        "medium": HealthcareMedium,
        "heavy": HealthcareHeavy,
    },

    "apple_quality": {
        "light": AppleQualityLight,
        "medium": AppleQualityMedium,
        "heavy": AppleQualityHeavy,
    }
}
