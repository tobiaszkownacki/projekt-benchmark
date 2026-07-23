from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

class StudentsPerformanceDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "Student_performance_data _.csv"
        df = pd.read_csv(filepath, sep=',')

        X = df.drop(columns=['GradeClass', 'StudentID'])
        y = df['GradeClass']

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])