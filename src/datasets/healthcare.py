from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

from sklearn.preprocessing import LabelEncoder

class HealthcareDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "healthcare.csv"
        df = pd.read_csv(filepath, sep=',')

        label_encoder = LabelEncoder()

        df['Gender'] = label_encoder.fit_transform(df['Gender'])
        df['Test Results'] = label_encoder.fit_transform(df['Test Results'])

        one_hot_columns = ['Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']

        df = pd.get_dummies(df, columns=one_hot_columns, dtype=int)

        X = df.drop(columns=['Test Results', 'Name', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date'])
        y = df['Test Results']

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])