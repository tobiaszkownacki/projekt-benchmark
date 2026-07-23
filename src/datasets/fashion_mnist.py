from typing import override

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

class FashionMNISTDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        train_set = torchvision.datasets.FashionMNIST(
            root=RAW_DATA_DIR / "f-mnist",
            train=True,
            download=True,
            transform=data_transforms,
        )

        test_set = torchvision.datasets.FashionMNIST(
            root=RAW_DATA_DIR / "f-mnist",
            train=False,
            download=True,
            transform=data_transforms,
        )

        return ConcatDataset([train_set, test_set])