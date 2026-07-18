from typing import override

import pandas as pd 
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import ConcatDataset

from sklearn.preprocessing import LabelEncoder

from src.config import RAW_DATA_DIR
from src.datasets.dataset import Dataset

class AiStudentImpactDataset(Dataset):
    @override
    def get(self) -> ConcatDataset:
        filepath = RAW_DATA_DIR / "ai_student_impact.csv"
        df = pd.read_csv(filepath, sep=',')

        label_encoder = LabelEncoder()
        one_hot_columns = ['Major_Category', 'Year_of_Study', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Paid_Subscription', 'Institutional_Policy']

        X = df.drop(columns=['Burnout_Risk_Level', 'Student_ID'])
        X = pd.get_dummies(X, columns=one_hot_columns, dtype=int)
        df['Burnout_Risk_Level'] = label_encoder.fit_transform(df['Burnout_Risk_Level'])
        y = df['Burnout_Risk_Level']

        print(X.shape[1])

        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.astype(int).values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return ConcatDataset([train_dataset])