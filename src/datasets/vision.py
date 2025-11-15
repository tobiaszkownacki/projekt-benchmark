"""Computer Vision datasets"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, TensorDataset
from sklearn.datasets import load_digits

from src.config import RAW_DATA_DIR
from src.datasets.base import BaseDatasetFactory


class VisionDatasetFactory(BaseDatasetFactory):
    """Factory for computer vision datasets"""
    
    @classmethod
    def get_data_set(cls, dataset_name: str) -> ConcatDataset:
        match dataset_name:
            case "cifar10":
                return cls._get_cifar10_data_set()
            case "digits":
                return cls._get_digits_data_set()
            case _:
                raise ValueError(f"Unsupported vision dataset: {dataset_name}")
    
    @classmethod
    def get_supported_datasets(cls) -> list[str]:
        return ["cifar10", "digits"]
    
    @classmethod
    def _get_cifar10_data_set(cls) -> ConcatDataset:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop(32, padding=4),
        ])
        
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=RAW_DATA_DIR / "cifar10",
            train=True,
            download=True,
            transform=train_transforms,
        )
        
        val_set = torchvision.datasets.CIFAR10(
            root=RAW_DATA_DIR / "cifar10",
            train=False,
            download=True,
            transform=val_transforms,
        )
        
        return train_set
    
    @classmethod
    def _get_digits_data_set(cls) -> ConcatDataset:
        digits = load_digits()
        X = digits.data
        y = digits.target

        X = X.reshape(-1, 1, 8, 8)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        X_tensor = X_tensor / 16.0  # digits data is originally in range [0, 16]

        full_dataset = TensorDataset(X_tensor, y_tensor)\
        
        return full_dataset
