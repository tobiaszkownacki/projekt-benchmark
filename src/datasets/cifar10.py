from typing import override

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset


class Cifar10Dataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop(32, padding=4),
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
