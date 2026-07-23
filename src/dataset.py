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

from models.light.ai_student_impact_light import AiStudentImpactLight
from models.medium.ai_student_impact_medium import AiStudentImpactMedium
from models.heavy.ai_student_impact_heavy import AiStudentImpactHeavy

from models.light.airplane_satisfaction_light import AirplaneSatisfactionfLight
from models.medium.airplane_satisfaction_medium import AirplaneSatisfactionMedium
from models.heavy.airplane_satisfaction_heavy import AirplaneSatisfactionfHeavy

from models.light.credit_score_light import CreditScoreLight
from models.medium.credit_score_medium import CreditScoreMedium
from models.heavy.credit_score_heavy import CreditScoreHeavy

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
from src.datasets.ai_student_impact import AiStudentImpactDataset
from src.datasets.airplane_satisfaction import AirplaneSatisfactionDataset
from src.datasets.credit_score import CreditScoreDataset


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
            case "ai_student_impact":
                return AiStudentImpactDataset().get()
            case "airplane_satisfaction":
                return AirplaneSatisfactionDataset().get()
            case "credit_score":
                return CreditScoreDataset().get()
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
    },
    "ai_student_impact": {
        "data_set": lambda: DataSetFactory.get_data_set("ai_student_impact")
    },
    "airplane_satisfaction": {
        "data_set": lambda: DataSetFactory.get_data_set("airplane_satisfaction")
    },
    "credit_score": {
        "data_set": lambda: DataSetFactory.get_data_set("credit_score")
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
    },

    "ai_student_impact": {
        "light": AiStudentImpactLight,
        "medium": AiStudentImpactMedium,
        "heavy": AiStudentImpactHeavy,
    },

    "airplane_satisfaction": {
        "light": AirplaneSatisfactionfLight,
        "medium": AirplaneSatisfactionMedium,
        "heavy": AirplaneSatisfactionfHeavy,
    },

    "credit_score": {
        "light": CreditScoreLight,
        "medium": CreditScoreMedium,
        "heavy": CreditScoreHeavy,
    }
}
