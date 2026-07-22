from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

from sklearn.preprocessing import LabelEncoder

class AirplaneSatisfactionDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "airplane_satisfaction.csv"
        df = pd.read_csv(filepath, sep=',')

        label_encoder = LabelEncoder()

        category_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

        df['satisfaction'] = label_encoder.fit_transform(df['satisfaction'])
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
        
        df = pd.get_dummies(df, columns=category_columns, dtype=int)

        X = df.drop(columns=['cid', 'id', 'satisfaction'])
        y = df['satisfaction']

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])