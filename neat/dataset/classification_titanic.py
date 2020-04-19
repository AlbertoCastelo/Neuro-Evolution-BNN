import os

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neat.dataset.abstract import NeatTestingDataset


class TitanicDataset(NeatTestingDataset):
    COLUMNS_USED = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    MEAN = [2.23669468,  0.36554622, 29.69911765,  0.51260504,  0.43137255, 34.69451401]
    STD = [0.83766265,  0.48158299, 14.51632115,  0.92913212,  0.85269161, 52.88185844]

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0):
        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type, random_state=random_state,
                         noise=noise)

    def generate_data(self):
        filename = ''.join([os.path.dirname(os.path.realpath(__file__)), '/data/titanic/train.csv'])
        data = pd.read_csv(filename)
        x = data.loc[:, self.COLUMNS_USED]
        x.dropna(axis=0, inplace=True)
        x.loc[x['Sex'] == 'male', 'Sex'] = 0
        x.loc[x['Sex'] == 'female', 'Sex'] = 1
        self.y = x['Survived'].values
        x.drop('Survived', axis=1, inplace=True)
        self.x_original = x.values

        self.x_original = self._add_noise(x=self.x_original)

        self.input_scaler = StandardScaler()
        self.input_scaler.fit(self.x_original)
        self.x = self.input_scaler.transform(self.x_original)

        self.x = torch.tensor(self.x).float()
        self.y = torch.tensor(self.y).long()

        self._generate_train_test_sets()

    def __len__(self):
        return len(self.x)
