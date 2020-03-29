import os

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms

from neat.dataset.abstract import NeatTestingDataset


class TitanicDataset(NeatTestingDataset):
    COLUMNS_USED = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    MEAN = [2.23669468,  0.36554622, 29.69911765,  0.51260504,  0.43137255, 34.69451401]
    STD = [0.83766265,  0.48158299, 14.51632115,  0.92913212,  0.85269161, 52.88185844]

    def __init__(self, train_percentage, dataset_type='train', ):
        self.x = None
        self.y = None
        self.train_percentage = train_percentage
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(self.MEAN, self.STD)])
        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type)

    def generate_data(self):
        filename = ''.join([os.path.dirname(os.path.realpath(__file__)), '/data/titanic/train.csv'])
        data = pd.read_csv(filename)
        x = data.loc[:, self.COLUMNS_USED]
        x.dropna(axis=0, inplace=True)
        x.loc[x['Sex'] == 'male', 'Sex'] = 0
        x.loc[x['Sex'] == 'female', 'Sex'] = 1
        y_numpy = x['Survived'].values
        x.drop('Survived', axis=1, inplace=True)
        x_numpy = (x.values - self.MEAN) / self.STD

        # train_test_split(x_numpy, y_numpy, test_size=0.33, random_state=42)

        self.y = torch.tensor(y_numpy)
        self.x = torch.tensor(x_numpy)

        self.x = self.x.float()
        self.y = self.y.long()

        data_limit = self._get_data_limit()
        self.x_train = self.x[:data_limit]
        self.y_train = self.y[:data_limit]

        self.x_test = self.x[data_limit:]
        self.y_test = self.y[data_limit:]

    def __len__(self):
        return len(self.x)
