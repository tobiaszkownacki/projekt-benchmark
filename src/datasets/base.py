from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset

class BaseDatasetFactory(ABC):
    """Abstract base class for dataset factories"""
    
    @classmethod
    @abstractmethod
    def get_data_set(cls, dataset_name: str) -> ConcatDataset:
        """
        Load and return train and validation datasets
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            ConcatDataset: dataset fot training
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_supported_datasets(cls) -> list[str]:
        """Return list of supported dataset names"""
        pass