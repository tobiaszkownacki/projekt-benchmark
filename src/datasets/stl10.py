from typing import override

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

class STL10Dataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_set = torchvision.datasets.STL10(
            root=RAW_DATA_DIR / "stl10",
            split='train',
            download=True,
            transform=data_transforms,
        )

        test_set = torchvision.datasets.STL10(
            root=RAW_DATA_DIR / "stl10",
            split='test',
            download=True,
            transform=data_transforms,
        )

        return ConcatDataset([train_set, test_set])