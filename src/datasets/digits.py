from typing import override

import torch
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.datasets.dataset import Dataset


class DigitsDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        digits = load_digits()
        X = digits.data
        y = digits.target

        X = X.reshape(-1, 1, 8, 8)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # digits data is originally in range [0, 16]
        X_tensor = X_tensor / 16.0

        full_dataset = TensorDataset(X_tensor, y_tensor)
        return ConcatDataset([full_dataset])
