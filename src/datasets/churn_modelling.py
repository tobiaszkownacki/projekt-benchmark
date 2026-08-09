from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

from sklearn.preprocessing import LabelEncoder

class ChurnModellingDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "Churn_Modelling.csv"
        df = pd.read_csv(filepath, sep=',')

        label_encoder = LabelEncoder()

        df['Gender'] = label_encoder.fit_transform(df['Gender'])
        df = pd.get_dummies(df, columns=['Geography'], dtype=int)

        X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
        y = df['Exited']

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])