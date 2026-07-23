from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

from sklearn.preprocessing import LabelEncoder

class CreditScoreDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "credit_score.csv"
        df = pd.read_csv(filepath, sep=',')

        df = df.drop(['ID','Customer_ID','Name','SSN'], axis=1)

        clean_num_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 
                             'Num_Credit_Inquiries', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']
        clean_cat_columns = ['Type_of_Loan', 'Credit_History_Age']

        for col in clean_num_columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

        for col in clean_cat_columns:
            df[col] = df[col].fillna('?')

        label_encoder = LabelEncoder()

        label_cat_columns = ['Month', 'Occupation', 'Type_of_Loan', 'Credit_History_Age', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour',
                             'Credit_Score']

        for col in label_cat_columns:
            df[col] = label_encoder.fit_transform(df[col])

        X = df.drop('Credit_Score',axis=1)
        y = df['Credit_Score']

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])