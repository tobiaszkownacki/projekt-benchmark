from src.datasets.vision import VisionDatasetFactory
from src.datasets.tabular import TabularDatasetFactory
from src.datasets.nlp import NLPDatasetFactory
from torch.utils.data import ConcatDataset


class UnifiedDatasetFactory:
    """Unified factory that routes to appropriate specialized factory"""
    
    _FACTORIES = {
        "vision": VisionDatasetFactory,
        "tabular": TabularDatasetFactory,
        "nlp": NLPDatasetFactory,
    }
    
    _DATASET_TYPES = {
        # Vision
        "cifar10": "vision",
        "digits": "vision",
        # Tabular
        "heart_disease": "tabular",
        "wine_quality": "tabular",
        # NLP
        "wmt14": "nlp",
        "imdb": "nlp",
    }
    
    @classmethod
    def get_data_set(cls, dataset_name: str) -> ConcatDataset:
        """
        Get dataset by name
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            ConcatDataset: Combined train and validation dataset
        """
        if dataset_name not in cls._DATASET_TYPES:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls._DATASET_TYPES.keys())}"
            )
        
        dataset_type = cls._DATASET_TYPES[dataset_name]
        factory = cls._FACTORIES[dataset_type]
        
        return factory.get_data_set(dataset_name)
    
    @classmethod
    def get_supported_datasets(cls) -> dict[str, list[str]]:
        """Get all supported datasets grouped by type"""
        return {
            factory_name: factory.get_supported_datasets()
            for factory_name, factory in cls._FACTORIES.items()
        }
    
    @classmethod
    def get_dataset_type(cls, dataset_name: str) -> str:
        """Get type of dataset (vision/tabular/nlp)"""
        return cls._DATASET_TYPES.get(dataset_name, "unknown")

def _get_model_class(model_name: str):
    match model_name:
        case "cifar10":
            from models.cifar10 import Cifar10
            return Cifar10
        case "heart_disease":
            from models.heart_disease import HeartDisease
            return HeartDisease
        case "wine_quality":
            from models.wine_quality import WineQuality
            return WineQuality
        case "digits":
            from models.digits import Digits
            return Digits
        case _:
            return lambda: None


DATA_SETS = {
    "cifar10": {
        "data_set": lambda: UnifiedDatasetFactory.get_data_set("cifar10"),
        "model": lambda: _get_model_class("cifar10"),
    },
    "heart_disease": {
        "data_set": lambda: UnifiedDatasetFactory.get_data_set("heart_disease"),
        "model": lambda: _get_model_class("heart_disease"),
    },
    "wine_quality": {
        "data_set": lambda: UnifiedDatasetFactory.get_data_set("wine_quality"),
        "model": lambda: _get_model_class("wine_quality"),
    },
    "digits": {
        "data_set": lambda: UnifiedDatasetFactory.get_data_set("digits"),
        "model": lambda: _get_model_class("digits"),
    },
    "wmt14": {
        "data_set": lambda: UnifiedDatasetFactory.get_data_set("wmt14"),
        "model": lambda: None,
    },
    "imdb": {
        "data_set": lambda: UnifiedDatasetFactory.get_data_set("imdb"),
        "model": lambda: None,
    }
}

__all__ = [
    "VisionDatasetFactory",
    "TabularDatasetFactory", 
    "NLPDatasetFactory",
    "UnifiedDatasetFactory",
    "DATA_SETS"
]