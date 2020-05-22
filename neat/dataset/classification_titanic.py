import os
import random

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from neat.dataset.abstract import NeatTestingDataset


class TitanicDataset(NeatTestingDataset):
    COLUMNS_USED = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    MEAN = [2.23669468,  0.36554622, 29.69911765,  0.51260504,  0.43137255, 34.69451401]
    STD = [0.83766265,  0.48158299, 14.51632115,  0.92913212,  0.85269161, 52.88185844]

    def __init__(self, train_percentage, dataset_type='train', random_state=42, noise=0.0, label_noise=0.0):
        directory = os.path.dirname(os.path.realpath(__file__))
        original_filename = ''.join([directory, f'/data/titanic/train.csv'])

        filename_noisy = ''.join([directory, f'/data/titanic/train_{noise}.csv'])
        if os.path.isfile(filename_noisy):
            data = pd.read_csv(filename_noisy)
        else:
            # create noisy dataset and save it.
            data_original = pd.read_csv(original_filename)
            data = self._create_noisy_dataset(data_original, filename=filename_noisy, noise=noise)

        self.data = data
        super().__init__(train_percentage=train_percentage, dataset_type=dataset_type, random_state=random_state,
                         noise=noise,label_noise=label_noise)

    def _generate_data(self):
        x = self.data.loc[:, self.COLUMNS_USED]
        x.dropna(axis=0, inplace=True)

        self.y = x['Survived'].values
        x.drop('Survived', axis=1, inplace=True)
        self.x_original = x.values

        self.input_scaler = StandardScaler()
        self.input_scaler.fit(self.x_original)
        self.x = self.input_scaler.transform(self.x_original)

        self.x = torch.tensor(self.x).float()
        self.y = torch.tensor(self.y).long()

        # self._generate_train_test_sets()
        #
        # self._add_noise_to_train_labels(self.label_noise)

    def __len__(self):
        return len(self.x)

    def _create_noisy_dataset(self, data_original, filename, noise):
        data_original = data_original.loc[:, self.COLUMNS_USED]
        data_original.dropna(axis=0, inplace=True)
        data_original.loc[data_original['Sex'] == 'male', 'Sex'] = 0
        data_original.loc[data_original['Sex'] == 'female', 'Sex'] = 1
        y = data_original['Survived'].values
        x = data_original.drop(columns=['Survived'], axis=1)

        means = np.mean(x, 0)
        stds = np.std(x, 0)
        x_noisy = ((x - means) / stds +
                   np.random.normal(0, noise, size=x.shape)
                   + means) * stds

        data_noisy = pd.DataFrame(x_noisy, columns=data_original.columns[1:])
        data_noisy['Survived'] = y
        data_noisy.to_csv(filename, index=False)
        return data_noisy
