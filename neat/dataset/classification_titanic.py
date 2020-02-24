import os

import torch
import pandas as pd
from torchvision.transforms import transforms

from neat.dataset.abstract import NeatTestingDataset


class TitanicDataset(NeatTestingDataset):
    COLUMNS_USED = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    MEAN = [2.23669468,  0.36554622, 29.69911765,  0.51260504,  0.43137255, 34.69451401]
    STD = [0.83766265,  0.48158299, 14.51632115,  0.92913212,  0.85269161, 52.88185844]

    def __init__(self, train_percentage=0.8, dataset_type='train', ):
        self.x = None
        self.y = None
        self.train_percentage = train_percentage
        self.train = False
        if dataset_type == 'train':
            self.train = True

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(self.MEAN, self.STD)])

    def generate_data(self):
        filename = ''.join([os.path.dirname(os.path.realpath(__file__)), '/data/titanic/train.csv'])
        data = pd.read_csv(filename)
        x = data.loc[:, self.COLUMNS_USED]
        x.dropna(axis=0, inplace=True)
        x.loc[x['Sex'] == 'male', 'Sex'] = 0
        x.loc[x['Sex'] == 'female', 'Sex'] = 1

        self.y = torch.tensor(x['Survived'].values)
        x.drop('Survived', axis=1, inplace=True)
        self.x = torch.tensor((x.values - self.MEAN) / self.STD)

        self.x = self.x.float()
        self.y = self.y.long()
