from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

from sklearn.preprocessing import LabelEncoder

class AppleQualityDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "apple_quality.csv"
        df = pd.read_csv(filepath, sep=',')

        label_encoder = LabelEncoder()

        df['Quality'] = label_encoder.fit_transform(df['Quality'])

        X = df.drop(columns=['Quality', 'A_id'])
        y = df['Quality']

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        return ConcatDataset([train_dataset])